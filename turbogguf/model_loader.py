"""Model loading with memory management for large models.

Handles loading HuggingFace models on CPU (rotation doesn't need GPU)
with memory-efficient strategies for 70B+ parameter models.
"""

import torch
from pathlib import Path
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, PreTrainedModel

from turbogguf.arch import get_handler
from turbogguf.arch.base import ArchHandler


def _ensure_cpu(model: PreTrainedModel) -> None:
    """Ensure all model parameters are on CPU for rotation.

    Moves CUDA tensors to CPU. If meta tensors are found (from accelerate
    offloading), raises an error with guidance.
    """
    meta_count = sum(1 for p in model.parameters() if p.device.type == "meta")
    if meta_count > 0:
        raise RuntimeError(
            f"{meta_count} parameters are on meta device (not loaded). "
            f"Use --device-map cpu instead of auto, or increase --max-memory."
        )

    cuda_count = sum(1 for p in model.parameters() if p.device.type == "cuda")
    if cuda_count > 0:
        print(f"Moving {cuda_count} parameters from CUDA to CPU...")
        model.to("cpu")
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def _materialize_and_check_meta(model: PreTrainedModel) -> None:
    """After removing accelerate hooks, verify no tensors remain on meta.

    accelerate uses the meta device for two distinct purposes:
      1. CPU offload — param lives in a CPU state-dict blob, accessed via hook.
         After remove_hook_from_submodules materializes these, they end up on
         the real CPU (fine for rotation).
      2. Disk offload — param lives in an offload_folder file, paged in via
         hook.  After hook removal these stay on meta (unusable).

    Call this AFTER remove_hook_from_submodules.  Anything still on meta is
    disk-offloaded; rotation would silently no-op on it and save_pretrained
    would fail.  Abort with a clear message instead.
    """
    meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if meta:
        sample = ", ".join(meta[:3]) + (f" ... (+{len(meta) - 3} more)" if len(meta) > 3 else "")
        raise RuntimeError(
            f"{len(meta)} parameter(s) are still on meta device after hook removal — "
            f"these were disk-offloaded and rotation cannot modify them.\n"
            f"Affected: {sample}\n"
            f"Fix: use --device-map cpu to let the OS pagefile handle the overflow, "
            f"or increase --max-memory so CPU+GPU together cover the full model size."
        )


def load_model(
    model_id: str,
    dtype: torch.dtype = torch.float16,
    device_map: str = "cpu",
    max_memory: Optional[dict] = None,
    trust_remote_code: bool = False,
    offload_folder: Optional[str] = None,
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
        "dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory
    if offload_folder is not None:
        # Disk offload is used by accelerate when max_memory doesn't cover the
        # full model.  Note: rotation modifies weights in-place via Parameter
        # reassignment, which works for CPU/CUDA-resident tensors but NOT for
        # disk-offloaded weights (those are read-on-demand).  Keep disk
        # allocation just above the overflow — a few GB for a tight fit.
        from pathlib import Path as _P
        _P(offload_folder).mkdir(parents=True, exist_ok=True)
        load_kwargs["offload_folder"] = offload_folder
        load_kwargs["offload_state_dict"] = True

    # Try CausalLM first (works for most text models), fall back to AutoModel
    # for multimodal wrappers like Gemma4ForConditionalGeneration
    model = None
    last_error = None
    for auto_cls in [AutoModelForCausalLM, AutoModel]:
        try:
            print(f"Trying {auto_cls.__name__}...")
            model = auto_cls.from_pretrained(model_id, **load_kwargs)
            break
        except Exception as e:
            last_error = e
            print(f"  {auto_cls.__name__} failed: {type(e).__name__}: {e}")
            continue

    if model is None:
        raise ValueError(
            f"Could not load model '{model_id}'. Last error: {last_error}"
        )

    model.eval()

    # For device_map="cpu", verify everything landed on CPU.
    # For device_map="auto" / mixed, leave weights where accelerate placed them
    # so we don't OOM trying to move 60+ GB onto CPU alone.  Rotation ops
    # are device-aware and work on whichever device each weight lives on.
    if device_map == "cpu":
        _ensure_cpu(model)

    # Remove accelerate dispatch hooks so weights are directly accessible
    # on their assigned devices (no on-demand movement interceptors).
    # For CPU-offloaded weights, hook removal materializes them onto real CPU.
    # For disk-offloaded weights, they stay on meta — _materialize_and_check_meta
    # catches that.
    try:
        from accelerate.hooks import remove_hook_from_submodules
        remove_hook_from_submodules(model)
    except (ImportError, Exception):
        pass

    if device_map != "cpu":
        _materialize_and_check_meta(model)

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
