"""Architecture-specific handlers for model weight access."""

from turbogguf.arch.base import ArchHandler
from turbogguf.arch.llama import LlamaHandler
from turbogguf.arch.mistral import MistralHandler
from turbogguf.arch.qwen2 import Qwen2Handler

ARCH_REGISTRY = {
    "LlamaForCausalLM": LlamaHandler,
    "MistralForCausalLM": MistralHandler,
    "Qwen2ForCausalLM": Qwen2Handler,
}


def get_handler(model) -> ArchHandler:
    """Auto-detect architecture and return the correct handler."""
    class_name = type(model).__name__
    if class_name in ARCH_REGISTRY:
        return ARCH_REGISTRY[class_name]()
    # Fallback: try LLaMA-like structure (most common)
    try:
        handler = LlamaHandler()
        handler.get_layers(model)  # test access
        return handler
    except (AttributeError, TypeError):
        pass
    raise ValueError(
        f"Unsupported architecture: {class_name}. "
        f"Supported: {list(ARCH_REGISTRY.keys())}"
    )
