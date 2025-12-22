# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .qwen3_moe import (Qwen3MoeDecoderLayer,
                          get_spec_layer_idx_from_weight_name)
from .interfaces import SupportsPP
from .utils import maybe_prefix


class SharedHead(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(config.vocab_size,
                                   config.hidden_size,
                                   quant_config=quant_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class Qwen3MoEMultiTokenPredictorLayer(nn.Module):

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        enable_eplb = parallel_config.enable_eplb

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2,
                                 config.hidden_size,
                                 bias=False)
        self.shared_head = SharedHead(config=config, quant_config=quant_config)
        self.mtp_block = Qwen3MoeDecoderLayer(config=config,
                                              cache_config=cache_config,
                                              quant_config=quant_config,
                                              prefix=prefix,
                                              enable_eplb=enable_eplb)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # masking inputs at position 0, as not needed by MTP
        inputs_embeds[positions == 0] = 0
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1))

        hidden_states, residual = self.mtp_block(positions=positions,
                                                 hidden_states=hidden_states,
                                                 residual=None)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3MoEMultiTokenPredictor(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        if hasattr(config, "num_nextn_predict_layers"):
            self.mtp_start_layer_idx = config.num_hidden_layers
            self.num_mtp_layers = config.num_nextn_predict_layers
        else:
            self.mtp_start_layer_idx = 0
            self.num_mtp_layers = config.num_hidden_layers
        # to map the exact layer index from weights
        self.layers = torch.nn.ModuleDict({
            str(idx):
            Qwen3MoEMultiTokenPredictorLayer(vllm_config,
                                             f"{prefix}.layers.{idx}")
            for idx in range(self.mtp_start_layer_idx,
                             self.mtp_start_layer_idx + self.num_mtp_layers)
        })
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = (spec_step_idx % self.num_mtp_layers)
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = (spec_step_idx % self.num_mtp_layers)
        mtp_layer = self.layers[str(self.mtp_start_layer_idx +
                                    current_step_idx)]
        logits = self.logits_processor(mtp_layer.shared_head.head,
                                       mtp_layer.shared_head(hidden_states),
                                       sampling_metadata)
        return logits


class Qwen3MoEMTP(nn.Module, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = Qwen3MoEMultiTokenPredictor(vllm_config=vllm_config,
                                                 prefix=maybe_prefix(
                                                     prefix, "model"))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, hidden_states,
                                   inputs_embeds, spec_step_idx)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        spec_step_idx: int = 0,
    ) -> Optional[torch.Tensor]:
        return self.model.compute_logits(hidden_states, sampling_metadata,
                                         spec_step_idx)

    def is_special(self, name:str) -> bool:
        return name == "lm_head.weight" or name == "model.norm.weight" or name == "model.embed_tokens.weight"

    def repeat_weight(self, weights):
        head_weight = weights.pop("lm_head.weight", None)
        norm_weight = weights.pop("model.norm.weight", None)
        embed_weight = weights.pop("model.embed_tokens.weight", None)
        for idx in range(self.model.mtp_start_layer_idx, self.model.mtp_start_layer_idx + self.model.num_mtp_layers):
            norm_name = f"model.layers.{idx}.shared_head.norm.weight"
            head_name = f"model.layers.{idx}.shared_head.head.weight"
            embed_name = f"model.layers.{idx}.embed_tokens.weight"
            if norm_weight is not None:
                weights[norm_name] = norm_weight
            if head_weight is not None:
                weights[head_name] = head_weight
            if embed_weight is not None:
                weights[embed_name] = embed_weight
        return weights

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        weights = self.repeat_weight(dict(weights)).items()
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue
            name = self._rewrite_spec_layer_name(spec_layer, name)
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if (("mlp.experts." in name) and name not in params_dict):
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    # According to DeepSeek-V3 Technical Report, MTP modules
                    # shares embedding layer. We only load the first weights.
                    if (spec_layer != self.model.mtp_start_layer_idx
                            and ".layers" not in name):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """
        Rewrite the weight name to match the format of the original model.
        Add .mtp_block for modules in transformer layer block for spec layer
        and rename shared layer weights to be top level.
        """

        if "shared_head" in name:
            return name
        
        if "norm_before_output" in name:  
            name = name.replace("norm_before_output", "hnorm") 
        
        if "norm_after_embedding" in name:
            name = name.replace("norm_after_embedding", "enorm") 
        
        if ".proj." in name:
            name = name.replace(".proj.", ".eh_proj.") 
        spec_layer_weight_names = [
            "embed_tokens", "enorm", "hnorm", "eh_proj",
        ]
        
        shared_weight_names = ["embed_tokens"]
        spec_layer_weight = False
        shared_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                if weight_name in shared_weight_names:
                    shared_weight = True
                break
        if not spec_layer_weight:
            # treat rest weights as weights for transformer layer block
            name = name.replace(f"model.layers.{spec_layer}.",
                                f"model.layers.{spec_layer}.mtp_block.")
        elif shared_weight:
            # treat shared weights as top level weights
            name = name.replace(f"model.layers.{spec_layer}.", "model.")
        return name
