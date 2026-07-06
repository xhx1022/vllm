# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared, torch-free helpers for routed-experts capture."""

from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpecKind,
    get_kv_cache_spec_kind,
)

logger = init_logger(__name__)

# KV specs that can anchor routed-experts slot mapping.
_FULL_ATTENTION_KINDS = frozenset(
    {
        KVCacheSpecKind.FULL_ATTENTION,
        KVCacheSpecKind.MLA_ATTENTION,
        KVCacheSpecKind.SINK_FULL_ATTENTION,
    }
)


def require_full_attention_gid(kv_cache_config: KVCacheConfig) -> int:
    """Return the full-attention KV group used as the routing anchor.

    Raises:
        ValueError: The model has no full-attention KV group (pure
            sliding-window / Mamba models are unsupported).
    """
    full_attn_gids = [
        gid
        for gid, g in enumerate(kv_cache_config.kv_cache_groups)
        if get_kv_cache_spec_kind(g.kv_cache_spec) in _FULL_ATTENTION_KINDS
    ]
    if not full_attn_gids:
        raise ValueError(
            "enable_return_routed_experts requires at least one full-attention "
            "KV cache group; pure sliding-window / Mamba models are unsupported."
        )
    if len(full_attn_gids) > 1:
        logger.warning_once(
            "enable_return_routed_experts: %d full-attention KV cache groups "
            "%s; anchoring routing on group %d only. Routing for tokens whose "
            "KV lives in the other group(s) is not offloaded.",
            len(full_attn_gids),
            full_attn_gids,
            full_attn_gids[0],
        )
    return full_attn_gids[0]


def get_num_experts_per_tok(hf_config) -> int:
    """Resolve the per-token expert count from the HF config."""
    val = getattr(hf_config, "num_experts_per_tok", None)
    if val is None:
        val = getattr(hf_config, "top_k_experts", None)
    if val is None:
        raise ValueError(
            "Cannot determine num_experts_per_tok: HF config has neither "
            "'num_experts_per_tok' nor 'top_k_experts'"
        )
    return val


def get_num_experts(hf_config) -> int:
    """Resolve the global logical expert count from the HF config."""
    for key in ("num_experts", "n_routed_experts", "num_local_experts"):
        val = getattr(hf_config, key, None)
        if val is not None:
            return val
    raise ValueError(
        "Could not resolve num_experts from model config. "
        "Expected one of 'num_experts', 'n_routed_experts', "
        "or 'num_local_experts'."
    )


def routing_slot_shape_dtype(
    vllm_config, kv_cache_config: KVCacheConfig
) -> tuple[tuple[int, int, int], str]:
    """Return the shared routing slot-buffer shape and dtype."""

    attn_gid = require_full_attention_gid(kv_cache_config)
    block_size = kv_cache_config.kv_cache_groups[attn_gid].kv_cache_spec.block_size
    hf_config = vllm_config.model_config.hf_text_config
    num_layers = hf_config.num_hidden_layers
    top_k = get_num_experts_per_tok(hf_config)
    num_experts = get_num_experts(hf_config)
    max_num_slots = kv_cache_config.num_blocks * block_size
    dtype = "uint8" if num_experts <= 256 else "uint16"
    return (max_num_slots, num_layers, top_k), dtype


def routed_experts_output_rank() -> int:
    """Return the rank that writes routing into the shared slot buffer."""
    return 0
