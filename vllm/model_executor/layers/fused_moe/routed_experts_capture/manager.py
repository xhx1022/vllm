# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side routed-experts buffers and block maps."""

import contextlib
import logging
import mmap
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt

from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.routed_experts_capture.common import (
    get_num_experts,
    get_num_experts_per_tok,
    require_full_attention_gid,
    routing_slot_shape_dtype,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capture.shared_region import (
    SharedRoutingRegion,
    shared_routing_mmap_path,
)
from vllm.v1.kv_cache_interface import KVCacheConfig

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingConnectorMetadata,
    )

logger = logging.getLogger(__name__)


class FullAttnBlockMap(NamedTuple):
    """Maps anchor-group GPU blocks to their offloaded sub-block slots."""

    gpu_block_ids: np.ndarray  # GPU block id per moved block
    cpu_block_ids: np.ndarray  # offloaded block holding that block
    sub_offsets: np.ndarray  # sub-block index within the offloaded block

    @classmethod
    def concatenate(cls, maps: list["FullAttnBlockMap"]) -> "FullAttnBlockMap":
        """Merge per-job maps so all blocks move in one vectorized copy."""
        return cls(
            gpu_block_ids=np.concatenate([m.gpu_block_ids for m in maps]),
            cpu_block_ids=np.concatenate([m.cpu_block_ids for m in maps]),
            sub_offsets=np.concatenate([m.sub_offsets for m in maps]),
        )


def _cdiv(a: int, b: int) -> int:
    """Ceiling division of non-negative integers."""
    return -(-a // b)


def compute_full_attn_block_map(
    gpu_block_ids: np.ndarray,
    cpu_block_ids: np.ndarray,
    group_sizes: Sequence[int],
    block_indices: Sequence[int],
    attn_gid: int,
    block_size_factor: int,
    expected_num_groups: int | None = None,
) -> FullAttnBlockMap:
    """Map a KV transfer job's anchor-group blocks to offload rows.

    Args:
        gpu_block_ids: Group-major GPU block ids for the whole job.
        cpu_block_ids: Group-major offloaded block ids for the whole job.
        group_sizes: GPU block count per KV cache group.
        block_indices: Logical block index in GPU blocks of each group's
            first block.
        attn_gid: Full-attention anchor group index.
        block_size_factor: GPU blocks per offloaded block.
        expected_num_groups: If set, the KV-group count the job must span;
            mismatch signals a contract break.

    Returns:
        FullAttnBlockMap covering only the anchor group.

    Raises:
        RuntimeError: If the group-major flat-order contract is violated.
    """
    factor = block_size_factor
    # Match the worker's per-group offloaded block counts.
    cpu_counts = [
        _cdiv(int(gs) + int(block_indices[g]) % factor, factor)
        for g, gs in enumerate(group_sizes)
    ]
    if (
        (expected_num_groups is not None and len(group_sizes) != expected_num_groups)
        or sum(group_sizes) != len(gpu_block_ids)
        or sum(cpu_counts) != len(cpu_block_ids)
    ):
        raise RuntimeError(
            "routed-experts offload transfer violates the group-major "
            f"flat-order contract: group_sizes={list(group_sizes)}, "
            f"block_indices={list(block_indices)}, attn_gid={attn_gid}, "
            f"factor={factor}, len(gpu)={len(gpu_block_ids)}, "
            f"len(cpu)={len(cpu_block_ids)}, expected_cpu={sum(cpu_counts)}, "
            f"expected_num_groups={expected_num_groups}"
        )

    # GPU offset: anchor group's GPU blocks start after prior groups'.
    gpu_off = int(sum(group_sizes[:attn_gid]))
    n = int(group_sizes[attn_gid])
    gpu_local = np.asarray(gpu_block_ids[gpu_off : gpu_off + n])

    if n == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return FullAttnBlockMap(empty_i, empty_i, empty_i.copy())

    # CPU offset: prior groups consume their cdiv counts.
    cpu_off = sum(cpu_counts[:attn_gid])
    skip = int(block_indices[attn_gid]) % factor
    p = np.arange(n, dtype=np.int64)
    sub_offsets = (skip + p) % factor
    cpu_local_idx = (skip + p) // factor
    cpu_local = np.asarray(cpu_block_ids)[cpu_off + cpu_local_idx]
    return FullAttnBlockMap(gpu_local, cpu_local, sub_offsets)


def _mmap_zeroed(shape: tuple[int, ...], dtype: npt.DTypeLike) -> np.ndarray:
    """Allocate a demand-paged zero ndarray backed by anonymous mmap."""
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if nbytes == 0:
        return np.zeros(shape, dtype=dtype)
    buf = mmap.mmap(-1, nbytes)  # anonymous, demand-paged, zero-filled
    return np.frombuffer(buf, dtype=dtype).reshape(shape)


class RoutedExpertsManager:
    """Scheduler-side slot and offload buffers for routed experts."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        num_offload_blocks: int | None = None,
        block_size_factor: int = 1,
    ) -> None:
        # Same anchor group as the worker.
        self.attn_gid = require_full_attention_gid(kv_cache_config)
        # Expected group count in every offload transfer spec's group_sizes.
        self.num_kv_groups = len(kv_cache_config.kv_cache_groups)
        attn_group = kv_cache_config.kv_cache_groups[self.attn_gid]
        self.block_size = attn_group.kv_cache_spec.block_size
        if block_size_factor < 1:
            raise ValueError(f"block_size_factor must be >= 1, got {block_size_factor}")
        self.block_size_factor = block_size_factor

        # KV groups share the same physical block id pool.
        hf_config = vllm_config.model_config.hf_text_config
        num_experts = get_num_experts(hf_config)
        num_experts_per_tok = get_num_experts_per_tok(hf_config)
        self.num_layers = hf_config.num_hidden_layers
        self.num_experts_per_tok = num_experts_per_tok
        max_num_slots = kv_cache_config.num_blocks * self.block_size
        # uint8 covers expert ids 0..255.
        expert_id_dtype = np.uint8 if num_experts <= 256 else np.uint16
        self.expert_id_dtype = expert_id_dtype
        # Validate the shared-mmap layout used by scheduler and worker.
        slot_shape, slot_dtype = routing_slot_shape_dtype(vllm_config, kv_cache_config)
        expected_shape = (max_num_slots, self.num_layers, num_experts_per_tok)
        if slot_shape != expected_shape:
            raise RuntimeError(
                "routed-experts slot buffer layout mismatch: "
                f"routing_slot_shape_dtype gave {slot_shape}, manager sized "
                f"{expected_shape}. The worker writer derives its mmap from the "
                "same helper, so a divergence here means the shared /dev/shm "
                "buffer would be misinterpreted across processes."
            )
        self._slot_region = SharedRoutingRegion(
            path=shared_routing_mmap_path(
                vllm_config.instance_id,
                vllm_config.parallel_config.data_parallel_rank,
            ),
            shape=slot_shape,
            dtype=slot_dtype,
        )
        self.routed_experts_by_slot = self._slot_region.array
        # Block-major zero-copy view over the slot buffer.
        self._blocks_view = self.routed_experts_by_slot.reshape(
            kv_cache_config.num_blocks,
            self.block_size,
            self.num_layers,
            num_experts_per_tok,
        )
        # Indexed by offloaded block id, then sub-block within that block.
        self.routed_experts_by_cpu_block: np.ndarray | None = None
        if num_offload_blocks is not None:
            self.routed_experts_by_cpu_block = _mmap_zeroed(
                (
                    num_offload_blocks,
                    self.block_size_factor,
                    self.block_size,
                    self.num_layers,
                    num_experts_per_tok,
                ),
                dtype=expert_id_dtype,
            )
        logger.info(
            "RoutedExpertsManager CPU buffer: %.2f GB "
            "(slots=%d, layers=%d, top_k=%d, dtype=%s), "
            "offloaded routed experts: %.2f GB "
            "(cpu_blocks=%s, block_size_factor=%d)",
            self.routed_experts_by_slot.nbytes / 1e9,
            max_num_slots,
            self.num_layers,
            num_experts_per_tok,
            self.routed_experts_by_slot.dtype.name,
            self.routed_experts_by_cpu_block.nbytes / 1e9
            if self.routed_experts_by_cpu_block is not None
            else 0.0,
            num_offload_blocks,
            self.block_size_factor,
        )

    def shutdown(self) -> None:
        """Release the shared slot mmap."""
        region = getattr(self, "_slot_region", None)
        if region is not None:
            # Drop the ndarray view before closing the mmap it is backed by.
            self.routed_experts_by_slot = None  # type: ignore[assignment]
            self._blocks_view = None  # type: ignore[assignment]
            region.close()
            self._slot_region = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.shutdown()

    def _cpu_blocks(self) -> np.ndarray:
        """Return the offloaded-block buffer, or raise if absent."""
        if self.routed_experts_by_cpu_block is None:
            raise RuntimeError(
                "routed-experts offload buffer is not initialized "
                "but a KV offload transfer was observed"
            )
        return self.routed_experts_by_cpu_block

    def store_to_offload_blocks(self, block_map: FullAttnBlockMap) -> None:
        """Copy GPU block rows to offloaded sub-block rows."""
        cpu_blocks = self._cpu_blocks()
        if len(block_map.gpu_block_ids) == 0:
            return
        # Vectorized over the job's blocks.
        cpu_blocks[block_map.cpu_block_ids, block_map.sub_offsets] = self._blocks_view[
            block_map.gpu_block_ids
        ]

    def load_from_offload_blocks(self, block_map: FullAttnBlockMap) -> None:
        """Copy offloaded sub-block rows to GPU block rows."""
        cpu_blocks = self._cpu_blocks()
        if len(block_map.gpu_block_ids) == 0:
            return
        self._blocks_view[block_map.gpu_block_ids] = cpu_blocks[
            block_map.cpu_block_ids, block_map.sub_offsets
        ]

    def _full_attn_block_map(
        self, gpu_spec: object, cpu_spec: object
    ) -> FullAttnBlockMap:
        """Map one KV offload transfer job to the routed-experts block map."""
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

        if not isinstance(gpu_spec, GPULoadStoreSpec):
            raise RuntimeError(
                f"expected GPULoadStoreSpec, got {type(gpu_spec).__name__}"
            )
        if not isinstance(cpu_spec, CPULoadStoreSpec):
            raise RuntimeError(
                f"expected CPULoadStoreSpec, got {type(cpu_spec).__name__}"
            )
        return compute_full_attn_block_map(
            gpu_block_ids=gpu_spec.block_ids,
            cpu_block_ids=cpu_spec.block_ids,
            group_sizes=gpu_spec.group_sizes,
            block_indices=gpu_spec.block_indices,
            attn_gid=self.attn_gid,
            block_size_factor=self.block_size_factor,
            expected_num_groups=self.num_kv_groups,
        )

    def apply_offload_transfers(
        self, meta: "OffloadingConnectorMetadata | None"
    ) -> None:
        """Store/load offloaded routing alongside this step's KV offload jobs.

        Runs after the worker writes this step's slots and before request
        outputs read routing back. Store rows are written as soon as
        prepare_store assigns block ids; loads are still gated by KV
        complete_store. ``meta`` is the step's kv_connector_metadata; the
        caller guarantees a CPU OffloadingConnector, so it is an
        ``OffloadingConnectorMetadata``.
        """
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
            OffloadingConnectorMetadata,
        )

        if meta is None:
            return
        if not isinstance(meta, OffloadingConnectorMetadata):
            raise RuntimeError(
                f"expected OffloadingConnectorMetadata, got {type(meta).__name__}"
            )
        # Batch every job into one fancy-index per direction. Under heavy
        # offload the per-job numpy call overhead dominates the scheduler
        # thread, while the data volume moved is the same. Empty maps are
        # dropped so concatenate() never sees a zero-length job.
        load_maps = []
        for job in meta.load_jobs.values():
            src, dst = job.src_spec, job.dst_spec  # CPU -> GPU
            block_map = self._full_attn_block_map(dst, src)
            if len(block_map.gpu_block_ids):
                load_maps.append(block_map)
        if load_maps:
            self.load_from_offload_blocks(FullAttnBlockMap.concatenate(load_maps))
        store_maps = []
        for job in meta.store_jobs.values():
            src, dst = job.src_spec, job.dst_spec  # GPU -> CPU
            block_map = self._full_attn_block_map(src, dst)
            if len(block_map.gpu_block_ids):
                store_maps.append(block_map)
        if store_maps:
            self.store_to_offload_blocks(FullAttnBlockMap.concatenate(store_maps))

    def read_cpu_blocks(self, cpu_block_ids: np.ndarray) -> np.ndarray:
        """Copy whole offloaded-block rows for a secondary store."""
        return self._cpu_blocks()[cpu_block_ids]

    def write_cpu_blocks(self, cpu_block_ids: np.ndarray, rows: np.ndarray) -> None:
        """Write whole offloaded-block rows loaded back from a secondary store."""
        self._cpu_blocks()[cpu_block_ids] = rows

    def get(
        self,
        block_ids: list[int],
        num_tokens: int,
        token_start: int = 0,
    ) -> np.ndarray:
        """Read routed-experts rows for a request token range.

        Args:
            block_ids: Block IDs from the attention KV-cache group.
            num_tokens: Number of tokens with routing rows.
            token_start: Skip the first token_start tokens.

        Returns:
            Array of shape (num_tokens - token_start, num_layers,
            num_experts_per_tok).
        """
        bs = self.block_size
        block_ids_array = np.asarray(block_ids, dtype=np.int64)
        # Avoid materializing the full (num_blocks, block_size) slot grid.
        pos = np.arange(token_start, num_tokens)
        slot_mapping = block_ids_array[pos // bs] * bs + (pos % bs)
        return self.routed_experts_by_slot[slot_mapping]

    def get_by_slots(self, slots: np.ndarray) -> np.ndarray:
        """Read routing for explicit slot indices (decode path)."""
        return self.routed_experts_by_slot[slots]
