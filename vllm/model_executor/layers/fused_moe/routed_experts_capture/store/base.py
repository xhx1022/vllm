# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend interface for routed-experts secondary-tier sidecars."""

import importlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec


class RoutedExpertsSecondaryStore(ABC):
    """Stores routing rows for KV blocks that have been offloaded to disk."""

    @abstractmethod
    def put(self, keys: Sequence[bytes], rows: np.ndarray) -> None:
        """Save each key's routing row. keys[i] maps to rows[i]."""

    @abstractmethod
    def get(self, keys: Sequence[bytes]) -> np.ndarray | None:
        """Load the rows for keys (same order), or None if any key is missing."""

    def prefetch(self, keys: Sequence[bytes]) -> None:  # noqa: B027
        """Optionally warm the cache so a following get() can skip the disk read."""


class RoutedExpertsStoreContext(NamedTuple):
    """Inputs for a secondary routing-store builder."""

    tier_config: dict
    offloading_spec: "OffloadingSpec"
    row_shape: tuple[int, ...]
    dtype: np.dtype


class RoutedExpertsStoreFactory:
    """Registry mapping secondary-tier types to routing-store builders."""

    _registry: dict[str, tuple[str, str]] = {}

    @classmethod
    def register_store(
        cls, tier_type: str, module_path: str, factory_name: str
    ) -> None:
        """Register a store-builder factory for a secondary-tier type."""
        if tier_type in cls._registry:
            raise ValueError(
                f"Routed-experts store for tier '{tier_type}' is already registered."
            )
        cls._registry[tier_type] = (module_path, factory_name)

    @classmethod
    def create(
        cls, tier_type: str, ctx: RoutedExpertsStoreContext
    ) -> RoutedExpertsSecondaryStore | None:
        """Build the store for tier_type, or None if no builder is known."""
        entry = cls._registry.get(tier_type)
        if entry is None:
            return None
        module_path, factory_name = entry
        module = importlib.import_module(module_path)
        builder = getattr(module, factory_name)
        return builder(ctx)
