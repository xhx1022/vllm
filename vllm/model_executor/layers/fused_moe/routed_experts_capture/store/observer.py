# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bridge KV-tier lifecycle events into a secondary routing store."""

import logging
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from vllm.model_executor.layers.fused_moe.routed_experts_capture.store.base import (
    RoutedExpertsSecondaryStore,
)
from vllm.v1.kv_offload.base import BlockLifecycleObserver

logger = logging.getLogger(__name__)


class _OffloadBuffer(Protocol):
    """Subset of RoutedExpertsManager used by the observer."""

    def read_cpu_blocks(self, cpu_block_ids: np.ndarray) -> np.ndarray: ...

    def write_cpu_blocks(self, cpu_block_ids: np.ndarray, rows: np.ndarray) -> None: ...


class RoutedExpertsBlockLifecycleObserver(BlockLifecycleObserver):
    """Mirror cascade / promotion events into a secondary routing store."""

    def __init__(
        self,
        manager: _OffloadBuffer,
        store: RoutedExpertsSecondaryStore,
    ) -> None:
        self._manager = manager
        self._store = store
        # Cumulative counters for observability.
        self.cascaded_blocks = 0
        self.promoted_blocks = 0

    def on_blocks_cascaded(
        self, keys: Sequence[bytes], cpu_block_ids: np.ndarray
    ) -> None:
        if len(keys) == 0:
            return
        rows = self._manager.read_cpu_blocks(np.asarray(cpu_block_ids))
        self._store.put(keys, rows)
        self.cascaded_blocks += len(keys)
        logger.debug(
            "routed-experts offload: cascaded %d block(s) to secondary (total=%d)",
            len(keys),
            self.cascaded_blocks,
        )

    def on_blocks_promotion_started(
        self, keys: Sequence[bytes], cpu_block_ids: np.ndarray
    ) -> None:
        # Warm the routing read-ahead cache while KV bytes promote.
        if len(keys) == 0:
            return
        self._store.prefetch(keys)

    def on_blocks_promoted(
        self, keys: Sequence[bytes], cpu_block_ids: np.ndarray
    ) -> None:
        if len(keys) == 0:
            return
        rows = self._store.get(keys)
        if rows is None:
            # Fail closed rather than leave stale offload-buffer rows.
            raise RuntimeError(
                f"routed-experts sidecar missing for {len(keys)} promoted "
                "block(s); KV was promoted but its routing rows are absent"
            )
        self._manager.write_cpu_blocks(np.asarray(cpu_block_ids), rows)
        self.promoted_blocks += len(keys)
        logger.debug(
            "routed-experts offload: promoted %d block(s) from secondary (total=%d)",
            len(keys),
            self.promoted_blocks,
        )

    def shutdown(self) -> None:
        """Flush the secondary store's pending writes, if it has any."""
        store_shutdown = getattr(self._store, "shutdown", None)
        if callable(store_shutdown):
            store_shutdown()
