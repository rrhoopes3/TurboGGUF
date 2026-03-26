"""TurboGGUF CLI: rotate, quantize, and evaluate LLM models.

Usage:
    turbogguf rotate --model <HF_ID> --output <DIR> [--seed 42]
    turbogguf pipeline --model <HF_ID> --quant Q2_K --output <GGUF> --llama-cpp <DIR>
    turbogguf evaluate --gguf <FILE> --llama-perplexity <BIN>
    turbogguf info --model <HF_ID>
"""

import os
import subprocess
import sys
from pathlib import Path

import click


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """TurboGGUF: Rotation-aware GGUF quantizer.

    Q2 quality that performs like Q4. No llama.cpp patching needed.
    """
    pass


@cli.command()
@click.option("--model", "-m", required=True, help="HuggingFace model ID or local path")
@click.option("--output", "-o", required=True, help="Output directory for rotated model")
@click.option("--seed", default=42, help="Random seed for Hadamard rotation")
@click.option("--no-r2", is_flag=True, help="Skip per-head R2 rotation")
@click.option("--dtype", default="float16", type=click.Choice(["float16", "bfloat16"]))
@click.option("--trust-remote-code", is_flag=True, help="Trust remote code for model loading")
def rotate(model, output, seed, no_r2, dtype, trust_remote_code):
    """Apply Hadamard rotation to model weights.

    Loads a HuggingFace model, fuses RMSNorm weights, applies R1+R2
    rotations, and saves as a standard HF checkpoint. The rotated model
    produces identical FP16 outputs but quantizes much better at low bit
    widths (Q2_K, Q3_K).
    """
    import torch
    from turbogguf.model_loader import load_model
    from turbogguf.rotation import rotate_model
    from turbogguf.export import export_rotated_model

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}

    click.echo(f"TurboGGUF v0.1.0 — Rotating {model}")
    click.echo(f"Seed: {seed}, R2: {'disabled' if no_r2 else 'enabled'}")
    click.echo()

    model_obj, tokenizer, handler = load_model(
        model,
        dtype=dtype_map[dtype],
        trust_remote_code=trust_remote_code,
    )

    metadata = rotate_model(
        model_obj,
        handler=handler,
        seed=seed,
        apply_r2=not no_r2,
    )

    export_rotated_model(model_obj, tokenizer, output, metadata=metadata)

    click.echo()
    click.echo(f"Rotated model saved to: {output}")
    click.echo("Next steps:")
    click.echo(f"  1. python convert_hf_to_gguf.py {output} --outtype f16")
    click.echo(f"  2. llama-quantize {output}/model.gguf output.gguf Q2_K")
    click.echo(f"  3. Load output.gguf in LM Studio")


@cli.command()
@click.option("--model", "-m", required=True, help="HuggingFace model ID or local path")
@click.option("--output", "-o", required=True, help="Output GGUF file path")
@click.option("--quant", "-q", default="Q2_K", help="Quantization type (Q2_K, Q3_K_S, Q4_K_M, etc.)")
@click.option("--seed", default=42, help="Random seed for Hadamard rotation")
@click.option("--llama-cpp", required=True, help="Path to llama.cpp directory")
@click.option("--no-r2", is_flag=True, help="Skip per-head R2 rotation")
@click.option("--trust-remote-code", is_flag=True, help="Trust remote code")
@click.option("--keep-intermediate", is_flag=True, help="Keep intermediate files")
def pipeline(model, output, quant, seed, llama_cpp, no_r2, trust_remote_code, keep_intermediate):
    """Full pipeline: rotate → convert to GGUF → quantize.

    One command to go from HuggingFace model to quantized GGUF.
    """
    import torch
    import tempfile
    import shutil
    from turbogguf.model_loader import load_model
    from turbogguf.rotation import rotate_model
    from turbogguf.export import export_rotated_model

    llama_cpp_dir = Path(llama_cpp)
    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    quantizer = llama_cpp_dir / "build" / "bin" / "llama-quantize"

    # Try alternative quantizer paths
    if not quantizer.exists():
        quantizer = llama_cpp_dir / "llama-quantize"
    if not quantizer.exists():
        quantizer = llama_cpp_dir / "build" / "llama-quantize"

    if not converter.exists():
        click.echo(f"Error: convert_hf_to_gguf.py not found at {converter}")
        sys.exit(1)
    if not quantizer.exists():
        click.echo(f"Error: llama-quantize not found. Tried multiple paths in {llama_cpp_dir}")
        sys.exit(1)

    # Step 1: Rotate
    click.echo("=" * 60)
    click.echo("STEP 1/3: Rotating model weights")
    click.echo("=" * 60)

    model_obj, tokenizer, handler = load_model(
        model, dtype=torch.float16, trust_remote_code=trust_remote_code,
    )
    metadata = rotate_model(model_obj, handler=handler, seed=seed, apply_r2=not no_r2)

    if keep_intermediate:
        rotated_dir = str(Path(output).parent / "rotated_intermediate")
    else:
        rotated_dir = tempfile.mkdtemp(prefix="turbogguf_")

    export_rotated_model(model_obj, tokenizer, rotated_dir, metadata=metadata)

    # Free model memory
    del model_obj
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 2: Convert to GGUF
    click.echo()
    click.echo("=" * 60)
    click.echo("STEP 2/3: Converting to GGUF (F16)")
    click.echo("=" * 60)

    f16_gguf = str(Path(rotated_dir) / "model.gguf")
    cmd_convert = [
        sys.executable, str(converter),
        rotated_dir,
        "--outtype", "f16",
        "--outfile", f16_gguf,
    ]
    result = subprocess.run(cmd_convert, capture_output=True, text=True)
    if result.returncode != 0:
        click.echo(f"Conversion failed:\n{result.stderr}")
        sys.exit(1)
    click.echo("GGUF conversion complete.")

    # Step 3: Quantize
    click.echo()
    click.echo("=" * 60)
    click.echo(f"STEP 3/3: Quantizing to {quant}")
    click.echo("=" * 60)

    cmd_quant = [str(quantizer), f16_gguf, output, quant]
    result = subprocess.run(cmd_quant, capture_output=True, text=True)
    if result.returncode != 0:
        click.echo(f"Quantization failed:\n{result.stderr}")
        sys.exit(1)

    # Cleanup
    if not keep_intermediate:
        shutil.rmtree(rotated_dir, ignore_errors=True)

    output_size = Path(output).stat().st_size / (1024**3)
    click.echo()
    click.echo("=" * 60)
    click.echo(f"Done! Output: {output} ({output_size:.2f} GB)")
    click.echo(f"Quant: {quant} with TurboGGUF rotation (seed={seed})")
    click.echo("Load in LM Studio — it's a standard GGUF file.")
    click.echo("=" * 60)


@cli.command()
@click.option("--gguf", "-g", required=True, multiple=True, help="GGUF file(s) to evaluate")
@click.option("--llama-perplexity", required=True, help="Path to llama-perplexity binary")
@click.option("--context-size", "-c", default=2048, help="Context window size")
@click.option("--output", "-o", help="Save results to JSON file")
def evaluate(gguf, llama_perplexity, context_size, output):
    """Evaluate and compare perplexity of GGUF models."""
    from turbogguf.evaluate import evaluate_gguf, compare_models, save_results

    results = []
    for path in gguf:
        label = Path(path).stem
        result = evaluate_gguf(
            path,
            llama_perplexity,
            context_size=context_size,
            label=label,
        )
        results.append(result)
        click.echo(f"  {result}")

    click.echo()
    click.echo(compare_models(results))

    if output:
        save_results(results, output)


@cli.command()
@click.option("--model", "-m", required=True, help="HuggingFace model ID")
def info(model):
    """Show memory requirements for rotating a model."""
    from turbogguf.model_loader import estimate_memory

    click.echo(f"Memory estimates for: {model}")
    click.echo()

    est = estimate_memory(model)
    click.echo(f"Estimated parameters:     {est['estimated_params_B']:.1f}B")
    click.echo(f"FP16 model size:          {est['fp16_gb']:.1f} GB")
    click.echo(f"Rotation overhead:        {est['rotation_overhead_gb']:.1f} GB")
    click.echo(f"Recommended system RAM:   {est['recommended_ram_gb']:.1f} GB")


if __name__ == "__main__":
    cli()
