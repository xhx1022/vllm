# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side routed-experts slot buffer."""

import contextlib
import logging
import numpy as np

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

logger = logging.getLogger(__name__)


class RoutedExpertsManager:
    """Scheduler-side slot buffer for routed experts."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        # Same anchor group as the worker.
        self.attn_gid = require_full_attention_gid(kv_cache_config)
        attn_group = kv_cache_config.kv_cache_groups[self.attn_gid]
        self.block_size = attn_group.kv_cache_spec.block_size

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
        logger.info(
            "RoutedExpertsManager slot buffer: %.2f GB "
            "(slots=%d, layers=%d, top_k=%d, dtype=%s)",
            self.routed_experts_by_slot.nbytes / 1e9,
            max_num_slots,
            self.num_layers,
            num_experts_per_tok,
            self.routed_experts_by_slot.dtype.name,
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
