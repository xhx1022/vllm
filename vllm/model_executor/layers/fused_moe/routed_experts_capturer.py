import logging
from abc import ABC
import torch
from vllm.config import ModelConfig

logger = logging.getLogger(__name__)

_global_experts_capturer = None

class RoutedExpertsCapturer(ABC):
    @staticmethod
    def create(enable: bool):
        """全局单例创建"""
        global _global_experts_capturer
        if _global_experts_capturer is not None:
            return _global_experts_capturer
        if enable:
            _global_experts_capturer = _RoutedExpertsCapturerReal()
        else:
            _global_experts_capturer = _RoutedExpertsCapturerNoop()
        return _global_experts_capturer

    @staticmethod
    def get_instance():
        if _global_experts_capturer is None:
            raise RuntimeError("Experts capturer not initialized.")
        return _global_experts_capturer

    def init_buffer(self, max_num_batched_tokens: int, model_config: ModelConfig):
        raise NotImplementedError

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        raise NotImplementedError

    def clear_buffer(self):
        raise NotImplementedError

    def get_captured_experts(self):
        raise NotImplementedError

    @staticmethod
    def is_real():
        return isinstance(_global_experts_capturer, _RoutedExpertsCapturerReal)
    @staticmethod
    def is_noop():
        return isinstance(_global_experts_capturer, _RoutedExpertsCapturerNoop)


class _RoutedExpertsCapturerReal(RoutedExpertsCapturer):
    """Capturer for routed experts with host buffer"""
    def __init__(self):
        self._experts_capturer_host_buffer = None

    def init_buffer(self, max_num_batched_tokens: int, model_config: ModelConfig):
        if (
            model_config.enable_return_routed_experts
            and self._experts_capturer_host_buffer is None
        ):
            self._experts_capturer_host_buffer = torch.zeros(
                (
                    model_config.hf_text_config.num_hidden_layers,
                    max_num_batched_tokens,
                    model_config.hf_text_config.num_experts_per_tok,
                ),
                dtype=torch.int32,
                device="cpu",
            )
            logger.debug(
                f"Initialized routed experts capturer host buffer with shape {self._experts_capturer_host_buffer.shape}."
            )

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        if self._experts_capturer_host_buffer is None:
            raise RuntimeError("Buffer not initialized.")
        batch_size, num_routed_experts = topk_ids.shape
        self._experts_capturer_host_buffer[layer_id, :batch_size, : ] = topk_ids.cpu()# to("cpu", non_blocking=True)

    def clear_buffer(self):
        if self._experts_capturer_host_buffer is not None:
            self._experts_capturer_host_buffer.zero_()

    def get_captured_experts(self):
        return self._experts_capturer_host_buffer


class _RoutedExpertsCapturerNoop(RoutedExpertsCapturer):
    def init_buffer(self, max_num_batched_tokens: int, model_config: ModelConfig):
        pass

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        pass

    def clear_buffer(self): 
        pass
    
    def get_captured_experts(self): 
        pass

