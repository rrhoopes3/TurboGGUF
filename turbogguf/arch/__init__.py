"""Architecture-specific handlers for model weight access."""

from turbogguf.arch.base import ArchHandler
from turbogguf.arch.llama import LlamaHandler
from turbogguf.arch.mistral import MistralHandler
from turbogguf.arch.qwen2 import Qwen2Handler
from turbogguf.arch.qwen3_moe import Qwen3MoeHandler
from turbogguf.arch.gemma import GemmaHandler, Gemma2Handler
from turbogguf.arch.gemma4 import Gemma4Handler

ARCH_REGISTRY = {
    "LlamaForCausalLM": LlamaHandler,
    "MistralForCausalLM": MistralHandler,
    "Qwen2ForCausalLM": Qwen2Handler,
    "Qwen3ForCausalLM": Qwen2Handler,  # dense Qwen3: same layout as Qwen2 sans qkv bias
    "Qwen3MoeForCausalLM": Qwen3MoeHandler,
    "Qwen3_5MoeForCausalLM": Qwen3MoeHandler,
    "Qwen3_5MoeForConditionalGeneration": Qwen3MoeHandler,
    "Qwen35MoeForCausalLM": Qwen3MoeHandler,
    "Qwen3_6MoeForCausalLM": Qwen3MoeHandler,
    "Qwen3_6MoeForConditionalGeneration": Qwen3MoeHandler,
    "GemmaForCausalLM": GemmaHandler,
    "Gemma2ForCausalLM": Gemma2Handler,
    "Gemma4ForConditionalGeneration": Gemma4Handler,
    "Gemma4ForCausalLM": Gemma4Handler,
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
