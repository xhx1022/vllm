# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManager: Pure-Python file system secondary tier for KV cache offloading.

Store path:
    Data is written to a temp file (<dest_path.tmp>) via os.write,
    then os.replace'd to the final path (without .tmp).

Load path:
    Data is read from the block file directly via os.readv into the
    provided memoryview slice.

File naming:  <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)
"""

import functools
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

try:
    from vllm.fs_io_C import batch_lookup as batch_lookup_C

    _HAS_BATCH_LOOKUP_C = True
except ImportError:
    _HAS_BATCH_LOOKUP_C = False

from typing_extensions import override

from vllm.distributed.kv_events import MEDIUM_FS
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingEvent,
    OffloadKey,
    ReqContext,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
    TierBlockBuffer,
)
from vllm.v1.kv_offload.tiering.fs.io import load_block, store_block
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


@dataclass(frozen=True)
class _BufferTransferSpec:
    view: memoryview
    block_size: int
    get_path: Callable[[OffloadKey], str]
    use_direct_io: bool


class FsAsyncLookupManager(AsyncLookupManager):
    """Async lookup manager for FileSystemTierManager."""

    def __init__(
        self,
        tier: "FileSystemTierManager",
        tier_type: str,
    ) -> None:
        super().__init__(tier_type=tier_type)
        self._tier = tier

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        buffer_specs = self._tier._buffer_specs
        if len(buffer_specs) == 1:
            paths = [buffer_specs[0].get_path(key) for key in keys]
            if _HAS_BATCH_LOOKUP_C:
                # C extension: GIL released for the entire faccessat() batch.
                return batch_lookup_C(paths)
            return (os.path.exists(path) for path in paths)
        # A block hits only if every sidecar file exists. Buffer-major lookup
        # keeps each key's results num_keys positions apart.
        num_keys = len(keys)
        paths = [
            buffer_spec.get_path(key) for buffer_spec in buffer_specs for key in keys
        ]
        if _HAS_BATCH_LOOKUP_C:
            path_exists = list(batch_lookup_C(paths))
        else:
            path_exists = [os.path.exists(path) for path in paths]
        return (
            all(
                path_exists[key_index + buffer_index * num_keys]
                for buffer_index in range(len(buffer_specs))
            )
            for key_index in range(num_keys)
        )


class FileSystemTierManager(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished_jobs() polls job completion and returns completed JobResults.

    Cross-process sharing:
        In order to enable KV cache sharing between multiple vLLM instances
        using the same ``root_dir`` (e.g., via a shared PVC) the environment
        variable ``PYTHONHASHSEED`` must be set to the same fixed value
        (e.g., "0") on all instances. Without this, each process initializes
        ``NONE_HASH`` (the chain-hash seed for block content hashes) with
        random bytes, producing different block filenames for identical token
        content.
    """

    medium: ClassVar[str] = MEDIUM_FS
    transfers_sidecar_buffers = True

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        enable_kv_events: bool = False,
    ):
        """
        Args:
            offloading_spec: contains the vllm_config, kv_cache_config
                and block_size_factor.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
            enable_kv_events: Emit BlockStored KV events for blocks
                successfully stored to this tier. Effective only when KV
                cache events are enabled globally (kv_events_config).
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)

        self.events: list[OffloadingEvent] | None = None
        if enable_kv_events:
            if offloading_spec.kv_events_config.enable_kv_cache_events:
                self.events = []
            else:
                logger.warning(
                    "enable_kv_events is set on secondary tier '%s' but KV "
                    "cache events are disabled globally; the tier will not "
                    "emit events.",
                    tier_type,
                )
        # Keys of in-flight store jobs, tracked only when events are enabled.
        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        # Opt in; FileMapper enables it only for a parallelism-invariant block.
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            gpu_blocks_per_file=offloading_spec.block_size_factor,
            parallel_agnostic=True,
        )

        # Write config file
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        self._lookup_manager = FsAsyncLookupManager(tier=self, tier_type=self.tier_type)

        # KV keeps O_DIRECT and its original file names; sidecar blocks are
        # not 512-byte aligned, so they use buffered I/O and a name suffix.
        self._buffer_specs: list[_BufferTransferSpec] = [
            _BufferTransferSpec(
                primary_kv_view,
                self._block_size,
                self.file_mapper.get_file_name,
                True,
            )
        ]

    @override
    def attach_primary_buffer(self, buffer: TierBlockBuffer) -> None:
        def get_path(key: OffloadKey, suffix: str = f".{buffer.name}") -> str:
            return self.file_mapper.get_file_name(key) + suffix

        self._buffer_specs.append(
            _BufferTransferSpec(buffer.view, buffer.block_size, get_path, False)
        )

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        return LookupResult.HIT if result else LookupResult.MISS

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = list(job_metadata.keys)
        # One task per (buffer, block); all buffers share the job_id, so the
        # job only completes once every buffer's blocks are on disk.
        tasks = (
            functools.partial(
                store_block,
                buffer_spec.get_path(key),
                buffer_spec.view,
                int(block_id) * buffer_spec.block_size,
                buffer_spec.block_size,
                buffer_spec.use_direct_io,
            )
            for buffer_spec in self._buffer_specs
            for key, block_id in zip(job_metadata.keys, job_metadata.block_ids)
        )
        num_tasks = len(job_metadata.keys) * len(self._buffer_specs)
        self._pool.enqueue_store(job_metadata.job_id, num_tasks, tasks)

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        tasks = (
            functools.partial(
                load_block,
                buffer_spec.get_path(key),
                buffer_spec.view,
                int(block_id) * buffer_spec.block_size,
                buffer_spec.block_size,
                buffer_spec.use_direct_io,
            )
            for buffer_spec in self._buffer_specs
            for key, block_id in zip(job_metadata.keys, job_metadata.block_ids)
        )
        num_tasks = len(job_metadata.keys) * len(self._buffer_specs)
        self._pool.enqueue_load(job_metadata.job_id, num_tasks, tasks)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Collect completed jobs from the finished-jobs queue.
        """
        results = []
        for job_id, success in self._pool.get_finished():
            if self.events is not None:
                keys = self._store_job_keys.pop(job_id, None)
                if success and keys:
                    self.events.append(
                        OffloadingEvent(keys=keys, medium=self.medium, removed=False)
                    )
            results.append(JobResult(job_id=job_id, success=success))
        return results

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    @override
    def drain_jobs(self) -> None:
        """Block until all in-flight transfers in the threadpool finish."""
        self._pool.wait_idle()

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_manager.flush()

    @override
    def shutdown(self) -> None:
        """
        Release resources held by this tier.

        Shuts down the lookup manager and the thread pool,
        clearing pending tasks and waiting for active threads to complete.
        """
        self._lookup_manager.shutdown()
        self._pool.shutdown(wait=True)
