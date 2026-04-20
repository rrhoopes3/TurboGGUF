"""Export rotated model as HuggingFace checkpoint.

After rotation, the model is saved in standard HF format (safetensors)
so it can be converted to GGUF using llama.cpp's convert_hf_to_gguf.py.
"""

import gc
import json
import os
import struct
from pathlib import Path
from typing import Optional

import torch.nn as nn
from transformers import PreTrainedTokenizer


def patch_gguf_output_tensor(gguf_path: str) -> bool:
    """In-place rename 'output.weight' → 'output' in a Gemma4 GGUF.

    Pre-built llama.cpp binaries ≤b8638 look for tensor named "output" (no
    ".weight" suffix) for Gemma4, because LLM_TENSOR_OUTPUT is absent from
    the Gemma4 arch tensor list in llama-arch.cpp.  When output.weight IS
    present in the GGUF (needed for correct gamma fusion), the binary fails
    with "wrong number of tensors: expected N, got N-1".

    This function patches only the GGUF *header* in-place (≤~100 KB), leaving
    the tensor data section completely untouched.  Total file size is unchanged:
    the 7-byte savings from shortening the name are added as extra alignment
    padding before the tensor data.

    Returns True if the patch was applied, False if not needed (wrong arch,
    tensor already named correctly, or tensor absent).
    """
    try:
        import numpy as np
        from gguf import GGUFReader
    except ImportError:
        raise RuntimeError(
            "gguf package required for GGUF patching. "
            "Install from llama.cpp/gguf-py or pip install gguf."
        )

    OLD = "output.weight"
    NEW = "output"

    # --- Read-only pass: check architecture and find tensor field offset ---
    reader = GGUFReader(str(gguf_path), mode='r')
    try:
        arch_field = reader.get_field('general.architecture')
        if arch_field is None or arch_field.contents() != 'gemma4':
            return False

        output_tensor = next((t for t in reader.tensors if t.name == OLD), None)
        if output_tensor is None:
            return False

        field_offset = output_tensor.field.offset
        data_offset = reader.data_offset
    finally:
        # Explicitly close mmap to release file lock on Windows
        reader.data._mmap.close()
        del reader
        gc.collect()

    # --- Read-write pass: in-place header patch ---
    old_entry_size = 8 + len(OLD)   # uint64 length (8) + "output.weight" (13) = 21
    new_entry_size = 8 + len(NEW)   # uint64 length (8) + "output" (6) = 14
    diff = old_entry_size - new_entry_size  # 7 bytes (header shrinks by this)

    suffix_start = field_offset + old_entry_size

    reader_rw = GGUFReader(str(gguf_path), mode='r+')
    try:
        # Verify we're patching the right bytes
        stored_len = struct.unpack_from('<Q', bytes(reader_rw.data[field_offset:field_offset + 8]))[0]
        stored_name = bytes(reader_rw.data[field_offset + 8:field_offset + 8 + stored_len])
        if stored_len != len(OLD) or stored_name != OLD.encode():
            raise ValueError(
                f"Unexpected tensor name bytes at offset {field_offset}: "
                f"len={stored_len}, name={stored_name}"
            )

        # Read the "suffix": everything from after OLD through data_offset.
        # This is: remaining tensor info entries + alignment padding.
        suffix = bytes(reader_rw.data[suffix_start:data_offset])

        # Build and write new name entry (14 bytes) at field_offset
        new_entry = struct.pack('<Q', len(NEW)) + NEW.encode()
        reader_rw.data[field_offset:field_offset + new_entry_size] = np.frombuffer(
            new_entry, dtype=np.uint8
        )

        # Shift suffix left by diff=7 bytes
        dest_start = field_offset + new_entry_size
        reader_rw.data[dest_start:dest_start + len(suffix)] = np.frombuffer(
            suffix, dtype=np.uint8
        )

        # Zero out the 7 bytes that are now extra alignment padding
        zero_start = dest_start + len(suffix)
        reader_rw.data[zero_start:zero_start + diff] = np.zeros(diff, dtype=np.uint8)

        reader_rw.data.flush()
    finally:
        reader_rw.data._mmap.close()
        del reader_rw
        gc.collect()

    print(f"  GGUF patched: '{OLD}' -> '{NEW}' (Gemma4 llama.cpp compatibility)")
    return True


def _is_text_decoder_norm_key(key: str) -> bool:
    """Match the text decoder's layer norms and final output norm only.

    Rotation only touches the text decoder's input_layernorm,
    post_attention_layernorm, pre/post_feedforward_layernorm, and the final
    model.(language_model.)?norm.weight.  Vision-tower / audio-tower norms
    are left untouched and must keep their learned gamma — skip them here
    and in `_force_norms_to_identity` so we don't corrupt the vision path
    and so verification doesn't flag them as failures.

    Also excludes per-head attention norms (q_norm / k_norm / v_norm) and
    DeltaNet's internal RMSNormGated (linear_attn.norm), whose gamma operates
    on head-level dimensions after the input projections and stays in the
    original basis across R1/R2.
    """
    if "vision_tower" in key or "audio_tower" in key or "visual" in key:
        return False
    if ".linear_attn." in key:
        # Qwen3.5/3.6 GatedDeltaNet internal per-head RMSNormGated — not a
        # residual-stream norm, left untouched by rotation.
        return False
    if key.endswith("_layernorm.weight"):
        return True
    if key.endswith(".norm.weight"):
        # Accept only the top-level `.norm.weight` (final output norm),
        # reject `.q_norm.weight` / `.k_norm.weight` / `.v_norm.weight`.
        tail = key.rsplit(".", 2)[-2]
        return tail == "norm"
    return False


def _norm_identity_value(model: nn.Module) -> float:
    """What value do identity-norm weights take, in the saved safetensors?

    For standard RMSNorm (LLaMA, Mistral, Qwen2, Qwen3 MoE plain):
        forward does `output * weight`, identity means weight = 1.
    For Gemma-style RMSNorm (Gemma, Qwen3.5 MoE, Qwen3.6 MoE):
        forward does `output * (1 + weight)`, identity means weight = 0.

    Detected via model_type on the text config — identity_value goes into
    both _force_norms_to_identity (what to overwrite to) and
    _verify_norms_are_identity (what deviation to measure against).
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return 1.0
    model_type = getattr(cfg, "model_type", "") or ""
    # Check text_config too for multimodal wrappers
    tc = getattr(cfg, "text_config", None)
    if tc is not None:
        model_type = getattr(tc, "model_type", "") or model_type
    gemma_style_prefixes = ("gemma", "qwen3_5_moe", "qwen3_6_moe", "qwen3_7_moe")
    if any(model_type.startswith(p) for p in gemma_style_prefixes):
        return 0.0
    return 1.0


def _force_norms_to_identity(output_path: Path, identity_value: float = 1.0) -> int:
    """Overwrite all *layernorm.weight and *.norm.weight tensors with 1.0.

    Workaround: with device_map="auto" + accelerate, in-place
    norm.weight.data.fill_(1.0) modifications during fuse_all_norms don't
    always persist through model.save_pretrained() — some norm tensors get
    written out with their original learned gamma values instead of identity.

    The rotation pipeline ALWAYS resets every norm to identity (gamma=1)
    after fusing its scale into adjacent linears.  This post-save pass
    enforces that invariant directly on the safetensors so the GGUF
    converter sees identity norms.

    Only touches keys ending in 'layernorm.weight' or 'norm.weight' to avoid
    overwriting unrelated parameters (q_norm/k_norm in some models are NOT
    layer norms in this sense — they end in '.q_norm.weight' /
    '.k_norm.weight' and could legitimately need scaling preserved; current
    handler treatment of those is architecture-specific).

    Returns the number of tensors patched.
    """
    import safetensors.torch as st
    import torch

    idx_path = output_path / "model.safetensors.index.json"
    if not idx_path.exists():
        return 0

    with open(idx_path) as f:
        idx = json.load(f)

    norm_keys = [k for k in idx["weight_map"] if _is_text_decoder_norm_key(k)]
    if not norm_keys:
        return 0

    # Group by shard so each shard is loaded/written exactly once
    shards: dict[str, list[str]] = {}
    for k in norm_keys:
        shards.setdefault(idx["weight_map"][k], []).append(k)

    patched = 0
    for shard_name, keys in shards.items():
        shard_path = output_path / shard_name
        tensors = st.load_file(str(shard_path))
        modified = False
        for k in keys:
            if k in tensors:
                if identity_value == 0.0:
                    tensors[k] = torch.zeros_like(tensors[k])
                else:
                    tensors[k] = torch.ones_like(tensors[k]) * identity_value
                patched += 1
                modified = True
        if modified:
            temp_path = output_path / f"_tmp_norm_{shard_name}"
            st.save_file(tensors, str(temp_path))
            os.remove(str(shard_path))
            os.rename(str(temp_path), str(shard_path))

    return patched


def _verify_norms_are_identity(
    output_path: Path,
    atol: float = 1e-4,
    identity_value: float = 1.0,
) -> list[tuple[str, float, float, float]]:
    """Scan saved safetensors and return norms whose weight is not all-ones.

    Returns list of (tensor_key, min, max, max_abs_deviation_from_1) for any
    norm tensor whose values deviate from 1.0 by more than atol.  An empty
    list means every layer/output norm is exactly identity, which is what
    the rotation pipeline requires for mathematical correctness.

    A non-empty list signals that either `reset_norm_to_identity` failed to
    persist for these norms OR the matching gamma fusion into adjacent
    linears also silently failed (with accelerate + device_map).  In that
    state the GGUF will produce gibberish at inference — abort instead.
    """
    import safetensors.torch as st
    import torch

    idx_path = output_path / "model.safetensors.index.json"
    if not idx_path.exists():
        # Single-shard case
        single = output_path / "model.safetensors"
        if not single.exists():
            return []
        tensors = st.load_file(str(single))
        shard_map = {k: "model.safetensors" for k in tensors.keys()}
        del tensors
    else:
        with open(idx_path) as f:
            shard_map = json.load(f)["weight_map"]

    norm_keys = [k for k in shard_map if _is_text_decoder_norm_key(k)]
    if not norm_keys:
        return []

    # Group by shard to load each shard once
    shards: dict[str, list[str]] = {}
    for k in norm_keys:
        shards.setdefault(shard_map[k], []).append(k)

    failures = []
    ones_cache: dict[tuple[int, torch.dtype], torch.Tensor] = {}
    for shard_name, keys in shards.items():
        tensors = st.load_file(str(output_path / shard_name))
        for k in keys:
            if k not in tensors:
                continue
            t = tensors[k].float()
            dev = float((t - identity_value).abs().max().item())
            if dev > atol:
                failures.append((k, float(t.min().item()), float(t.max().item()), dev))
        del tensors
    return failures


def _fix_lm_head_key_for_multimodal(output_path: Path) -> None:
    """Rename lm_head.weight for multimodal GGUF converter compatibility.

    llama.cpp's Gemma4 converter filters tensors by "language_model." prefix,
    dropping top-level lm_head.weight. This renames it to
    language_model.lm_head.weight so the converter:
      1. Passes the filter ("language_model." in name)
      2. Strips the prefix -> lm_head.weight
      3. Maps to output.weight in the GGUF
    """
    import safetensors.torch as st

    idx_path = output_path / "model.safetensors.index.json"
    if not idx_path.exists():
        return

    with open(idx_path) as f:
        idx = json.load(f)

    if "lm_head.weight" not in idx["weight_map"]:
        return

    shard_name = idx["weight_map"]["lm_head.weight"]
    shard_path = output_path / shard_name
    temp_path = output_path / f"_temp_{shard_name}"

    tensors = st.load_file(str(shard_path))
    if "lm_head.weight" in tensors:
        tensors["language_model.lm_head.weight"] = tensors.pop("lm_head.weight")
        st.save_file(tensors, str(temp_path))
        os.remove(str(shard_path))
        os.rename(str(temp_path), str(shard_path))

        idx["weight_map"]["language_model.lm_head.weight"] = idx["weight_map"].pop("lm_head.weight")
        with open(idx_path, "w") as f:
            json.dump(idx, f, indent=2)
        print("  Renamed lm_head.weight -> language_model.lm_head.weight for GGUF converter")


def export_rotated_model(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    metadata: Optional[dict] = None,
) -> str:
    """Save rotated model as a standard HuggingFace checkpoint.

    The saved model is a drop-in replacement — same architecture, same
    config, just with rotated weights. llama.cpp's converter and quantizer
    handle it without modification.

    Args:
        model: The rotated model
        tokenizer: Associated tokenizer
        output_dir: Directory to save to
        metadata: Rotation metadata to save as sidecar

    Returns:
        Path to the output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving rotated model to {output_path}...")

    # Save model weights (safetensors format)
    # Use small shards to avoid OOM on memory-constrained systems
    model.save_pretrained(
        str(output_path),
        safe_serialization=True,
        max_shard_size="4GB",
    )

    # Save tokenizer EARLY — before the norm-verification gate below.
    # If verification aborts, the intermediate directory still has a usable
    # tokenizer so downstream tools (convert_hf_to_gguf.py) and manual
    # retries don't have to re-download from HF just to get tokenizer files.
    tokenizer.save_pretrained(str(output_path))

    # Force all layer/output norms to identity (1.0). They were reset
    # in-memory by fuse_all_norms but device_map='auto' + accelerate can
    # silently drop those changes during save_pretrained for some norms.
    #
    # PRE-CHECK: verify whether the in-memory reset actually reached disk.
    # If it didn't, the matching gamma-into-linear fusion also silently
    # failed (same accelerate dispatch bug), so overwriting norms to 1.0
    # alone would leave mis-rotated linears — producing gibberish GGUF
    # (this is what happened with v7).  Abort instead of silently shipping.
    identity_value = _norm_identity_value(model)
    pre_failures = _verify_norms_are_identity(output_path, identity_value=identity_value)
    if pre_failures:
        msg = (
            f"ROTATION FAILED: {len(pre_failures)} norm tensors are not identity "
            f"({identity_value}) after save_pretrained.  This means "
            f"reset_norm_to_identity did not persist to disk (accelerate + "
            f"device_map dispatch bug), so the matching gamma-into-linear "
            f"fusions also did not persist.  The GGUF would produce gibberish "
            f"at inference.\n\n"
            f"First 10 offending norms (key, min, max, |gamma-identity|_max):\n"
        )
        for k, mn, mx, dev in pre_failures[:10]:
            msg += f"  {k}: min={mn:.4f} max={mx:.4f} dev={dev:.4f}\n"
        raise RuntimeError(msg)

    n_patched = _force_norms_to_identity(output_path, identity_value=identity_value)
    if n_patched:
        print(f"  Forced {n_patched} norm tensors to identity (gamma=1)")

    # For multimodal wrappers (e.g. Gemma4ForConditionalGeneration), the
    # GGUF converter expects all text model tensors to have "language_model."
    # prefix. Top-level lm_head.weight gets dropped without this fix.
    is_multimodal = hasattr(model, "model") and hasattr(model.model, "language_model")
    if is_multimodal:
        _fix_lm_head_key_for_multimodal(output_path)

    # (tokenizer was saved earlier, before the norm-verification gate)

    # Save rotation metadata as sidecar
    if metadata is not None:
        meta_path = output_path / "turbogguf_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Rotation metadata saved to {meta_path}")

    # Compute total size
    total_bytes = sum(
        f.stat().st_size
        for f in output_path.rglob("*")
        if f.is_file()
    )
    print(f"Saved: {total_bytes / (1024**3):.2f} GB")

    return str(output_path)
