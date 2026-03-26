"""Export rotated model as HuggingFace checkpoint.

After rotation, the model is saved in standard HF format (safetensors)
so it can be converted to GGUF using llama.cpp's convert_hf_to_gguf.py.
"""

import json
import os
from pathlib import Path
from typing import Optional

import torch.nn as nn
from transformers import PreTrainedTokenizer


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
    model.save_pretrained(
        str(output_path),
        safe_serialization=True,
    )

    # Save tokenizer
    tokenizer.save_pretrained(str(output_path))

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
