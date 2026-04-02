"""Model loading with memory management for large models.

Handles loading HuggingFace models on CPU (rotation doesn't need GPU)
with memory-efficient strategies for 70B+ parameter models.
"""

import torch
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, PreTrainedModel

from turbogguf.arch import get_handler
from turbogguf.arch.base import ArchHandler


def load_model(
    model_id: str,
    dtype: torch.dtype = torch.float16,
    device_map: str = "cpu",
    max_memory: Optional[dict] = None,
    trust_remote_code: bool = False,
) -> Tuple[PreTrainedModel, AutoTokenizer, ArchHandler]:
    """Load a HuggingFace model for rotation.

    Models are loaded on CPU by default since rotation is pure matrix math
    and doesn't benefit from GPU. For 70B+ models, use device_map="auto"
    with max_memory to shard across CPU RAM.

    Tries AutoModelForCausalLM first (standard text models), then falls back
    to AutoModel for multimodal wrappers (e.g., Gemma4ForConditionalGeneration).

    Args:
        model_id: HuggingFace model ID or local path
        dtype: Model dtype (float16 recommended for rotation)
        device_map: Device mapping strategy
        max_memory: Max memory per device (e.g., {"cpu": "48GB"})
        trust_remote_code: Whether to trust remote code

    Returns:
        Tuple of (model, tokenizer, architecture_handler)
    """
    print(f"Loading model: {model_id}")
    print(f"dtype: {dtype}, device_map: {device_map}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )

    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory

    # Try CausalLM first (works for most text models), fall back to AutoModel
    # for multimodal wrappers like Gemma4ForConditionalGeneration
    model = None
    for auto_cls in [AutoModelForCausalLM, AutoModel]:
        try:
            model = auto_cls.from_pretrained(model_id, **load_kwargs)
            break
        except (ValueError, KeyError):
            continue

    if model is None:
        raise ValueError(
            f"Could not load model '{model_id}' with AutoModelForCausalLM or AutoModel. "
            f"Check that the model ID is correct and transformers is up to date."
        )

    model.eval()

    handler = get_handler(model)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {param_count / 1e9:.1f}B parameters")
    print(f"Architecture: {type(model).__name__}")

    return model, tokenizer, handler


def estimate_memory(model_id: str) -> dict:
    """Estimate memory requirements without loading the model.

    Args:
        model_id: HuggingFace model ID

    Returns:
        Dict with memory estimates in GB
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id)

    # Rough parameter count estimation
    hidden = config.hidden_size
    layers = config.num_hidden_layers
    vocab = config.vocab_size
    intermediate = getattr(config, "intermediate_size", hidden * 4)

    # Each layer: q,k,v,o (4 * hidden^2) + gate,up,down (3 * hidden * intermediate)
    params_per_layer = 4 * hidden * hidden + 3 * hidden * intermediate
    total_params = layers * params_per_layer + vocab * hidden + hidden * vocab

    fp16_gb = total_params * 2 / (1024**3)
    fp32_gb = total_params * 4 / (1024**3)

    return {
        "estimated_params": total_params,
        "estimated_params_B": total_params / 1e9,
        "fp16_gb": fp16_gb,
        "fp32_gb": fp32_gb,
        "rotation_overhead_gb": fp32_gb * 0.1,  # ~10% overhead for rotation matrices
        "recommended_ram_gb": fp16_gb * 1.3,  # 30% headroom
    }
