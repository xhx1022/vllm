# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Filesystem secondary-tier sidecar for routed-experts rows."""

import functools
import logging
import os
import threading
import time
from collections.abc import Sequence

import numpy as np

from vllm.model_executor.layers.fused_moe.routed_experts_capture.store.base import (
    RoutedExpertsSecondaryStore,
    RoutedExpertsStoreContext,
    RoutedExpertsStoreFactory,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

logger = logging.getLogger(__name__)


class FileRoutedExpertsStore(RoutedExpertsSecondaryStore):
    """Disk-backed routed-experts secondary store."""

    def __init__(
        self,
        file_mapper,
        row_shape: tuple[int, ...],
        dtype: np.dtype,
        n_write_threads: int = 4,
        n_read_threads: int = 4,
    ) -> None:
        self._file_mapper = file_mapper
        self._row_shape = row_shape
        self._dtype = np.dtype(dtype)
        self._row_bytes = int(np.prod(row_shape)) * self._dtype.itemsize

        # Rows whose async write has not completed; preserves read-after-write.
        self._pending: dict[bytes, np.ndarray] = {}
        # Read-ahead cache filled by prefetch and consumed by get.
        self._prefetched: dict[bytes, np.ndarray] = {}
        self._cache_lock = threading.Lock()

        # Same dual-queue primitive used by the KV filesystem tier.
        self._job_counter = 0
        self._closed = False
        self._pool = DualQueueThreadPool(
            n_read_threads=max(1, n_read_threads),
            n_write_threads=max(1, n_write_threads),
            thread_name_prefix="vllm_re_fs",
        )

    def _path(self, key: bytes) -> str:
        # FileMapper.get_file_name returns "<...>/<hash>.bin"; swap the
        # suffix so routing sidecars never collide with KV block files.
        return self._file_mapper.get_file_name(key)[: -len(".bin")] + ".re"

    def _read_row(self, path: str) -> np.ndarray | None:
        """Read one sidecar off disk, or None if absent."""
        try:
            fd = os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            return None
        try:
            row = np.empty(self._row_shape, dtype=self._dtype)
            view = memoryview(row).cast("B")
            got = 0
            while got < self._row_bytes:
                chunk = os.readv(fd, [view[got:]])
                if chunk == 0:
                    break  # EOF before full row -> truncated
                got += chunk
        finally:
            os.close(fd)
        if got != self._row_bytes:
            raise RuntimeError(
                f"routed-experts sidecar {path} has {got} bytes, "
                f"expected {self._row_bytes}"
            )
        return row

    def _write_one(self, key: bytes, path: str, row: np.ndarray) -> None:
        """Write one sidecar atomically, then drop it from pending."""
        try:
            if os.path.exists(path):
                return
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            payload = memoryview(row).cast("B")
            try:
                fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                try:
                    written = 0
                    while written < self._row_bytes:
                        written += os.write(fd, payload[written:])
                finally:
                    os.close(fd)
                os.replace(tmp, path)
            except Exception:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise
        finally:
            with self._cache_lock:
                self._pending.pop(key, None)

    def _read_one(self, key: bytes, path: str) -> None:
        """Prefetch one sidecar into _prefetched."""
        try:
            row = self._read_row(path)
        except Exception as exc:
            logger.warning("routed-experts prefetch read failed (%s): %s", path, exc)
            return
        if row is not None:
            with self._cache_lock:
                self._prefetched[key] = row

    def _reap_finished(self) -> None:
        """Drain the pool's finished-jobs queue."""
        self._pool.get_finished()

    def _next_job_id(self) -> int:
        self._job_counter += 1
        return self._job_counter

    def put(self, keys: Sequence[bytes], rows: np.ndarray) -> None:
        self._reap_finished()
        tasks = []
        for key, row in zip(keys, rows):
            path = self._path(key)
            # Copy now: the source CPU block may be reused after put returns.
            row_copy = np.array(row, dtype=self._dtype, copy=True, order="C")
            with self._cache_lock:
                self._pending[key] = row_copy
            tasks.append(functools.partial(self._write_one, key, path, row_copy))
        if tasks:
            self._pool.enqueue_store(self._next_job_id(), len(tasks), tasks)

    def prefetch(self, keys: Sequence[bytes]) -> None:
        self._reap_finished()
        tasks = []
        for key in keys:
            # get will hit these from memory without touching disk.
            with self._cache_lock:
                if key in self._pending or key in self._prefetched:
                    continue
            tasks.append(functools.partial(self._read_one, key, self._path(key)))
        if tasks:
            self._pool.enqueue_load(self._next_job_id(), len(tasks), tasks)

    def get(self, keys: Sequence[bytes]) -> np.ndarray | None:
        self._reap_finished()
        rows = np.empty((len(keys), *self._row_shape), dtype=self._dtype)
        for i, key in enumerate(keys):
            # Pending preserves read-after-write; prefetched rows are consumed.
            with self._cache_lock:
                cached = self._pending.get(key)
                if cached is None:
                    cached = self._prefetched.pop(key, None)
            if cached is not None:
                rows[i] = cached
                continue
            row = self._read_row(self._path(key))
            if row is None:
                return None
            rows[i] = row
        return rows

    def shutdown(self, drain_timeout: float = 30.0) -> None:
        """Flush pending writes and stop the pool."""
        if self._closed:
            return
        self._closed = True

        waited, step = 0.0, 0.005
        while waited < drain_timeout:
            with self._cache_lock:
                if not self._pending:
                    break
            self._pool.get_finished()
            time.sleep(step)
            waited += step

        with self._cache_lock:
            leaked = len(self._pending)
        if leaked:
            logger.warning(
                "routed-experts store shutdown: %d sidecar write(s) did not "
                "drain within %.1fs; they may be missing on disk",
                leaked,
                drain_timeout,
            )
        self._pool.shutdown(wait=True)


def _fs_routed_experts_root(ctx: RoutedExpertsStoreContext) -> str:
    """Resolve the filesystem root for fs-backed routing sidecars."""
    import tempfile

    root_dir = ctx.tier_config.get("root_dir")
    if root_dir:
        return os.path.join(str(root_dir), "routed_experts")
    instance_id = ctx.offloading_spec.vllm_config.instance_id
    return os.path.join(tempfile.gettempdir(), f"vllm_routed_experts_{instance_id}")


def build_fs_routed_experts_store(
    ctx: RoutedExpertsStoreContext,
) -> RoutedExpertsSecondaryStore:
    """Builder for the built-in filesystem secondary tier (type="fs")."""
    from vllm.v1.kv_offload.file_mapper import FileMapper

    spec = ctx.offloading_spec
    root = _fs_routed_experts_root(ctx)
    file_mapper = FileMapper.from_offloading_spec(
        root_dir=root,
        offloading_spec=spec,
        gpu_blocks_per_file=spec.block_size_factor,
    )
    return FileRoutedExpertsStore(
        file_mapper=file_mapper,
        row_shape=ctx.row_shape,
        dtype=ctx.dtype,
        # Routing sidecars are much smaller than KV blocks.
        n_write_threads=int(ctx.tier_config.get("n_write_threads", 4)),
        n_read_threads=int(ctx.tier_config.get("n_read_threads", 4)),
    )


RoutedExpertsStoreFactory.register_store(
    "fs",
    "vllm.model_executor.layers.fused_moe.routed_experts_capture.store.fs",
    "build_fs_routed_experts_store",
)
