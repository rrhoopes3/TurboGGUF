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
@click.option("--quant", "-q", default="Q2_K",
              help="Quantization type (Q2_K, Q3_K_S, Q4_K_M, IQ2_XXS, IQ3_XXS, IQ4_XS, etc.)")
@click.option("--seed", default=42, help="Random seed for Hadamard rotation")
@click.option("--llama-cpp", required=True, help="Path to llama.cpp directory")
@click.option("--no-r2", is_flag=True, help="Skip per-head R2 rotation")
@click.option("--trust-remote-code", is_flag=True, help="Trust remote code")
@click.option("--keep-intermediate", is_flag=True, help="Keep intermediate files")
@click.option("--imatrix", is_flag=True,
              help="Generate importance matrix before quantization (recommended for IQ types)")
@click.option("--imatrix-dataset", type=click.Path(exists=True),
              help="Calibration dataset for imatrix (text file, one sample per line)")
@click.option("--imatrix-file", type=click.Path(),
              help="Pre-computed .imatrix file to use (skips generation)")
def pipeline(model, output, quant, seed, llama_cpp, no_r2, trust_remote_code,
             keep_intermediate, imatrix, imatrix_dataset, imatrix_file):
    """Full pipeline: rotate -> convert to GGUF -> [imatrix] -> quantize.

    One command to go from HuggingFace model to quantized GGUF.

    \b
    Standard K-quants:   Q2_K, Q3_K_S, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K
    I-quants (need --imatrix or --imatrix-file):
                         IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL

    I-quants combined with TurboGGUF rotation give the best quality-per-bit.
    Use --imatrix with a calibration dataset for optimal results.
    """
    import torch
    import tempfile
    import shutil
    from turbogguf.model_loader import load_model
    from turbogguf.rotation import rotate_model
    from turbogguf.export import export_rotated_model

    # I-quant types that require an importance matrix
    iq_types = {"IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_XS", "IQ4_NL"}
    needs_imatrix = quant.upper() in iq_types

    if needs_imatrix and not imatrix and not imatrix_file:
        click.echo(f"Warning: {quant} benefits significantly from --imatrix. "
                    f"Enabling imatrix generation automatically.")
        imatrix = True

    llama_cpp_dir = Path(llama_cpp)
    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    quantizer = llama_cpp_dir / "build" / "bin" / "llama-quantize"

    # Try alternative quantizer paths
    if not quantizer.exists():
        quantizer = llama_cpp_dir / "llama-quantize"
    if not quantizer.exists():
        quantizer = llama_cpp_dir / "build" / "llama-quantize"

    # Find imatrix binary if needed
    imatrix_bin = None
    if imatrix and not imatrix_file:
        for candidate in [
            llama_cpp_dir / "build" / "bin" / "llama-imatrix",
            llama_cpp_dir / "llama-imatrix",
            llama_cpp_dir / "build" / "llama-imatrix",
        ]:
            if candidate.exists():
                imatrix_bin = candidate
                break
        if imatrix_bin is None:
            click.echo("Error: llama-imatrix not found. Build llama.cpp with imatrix support,")
            click.echo("or provide a pre-computed file with --imatrix-file.")
            sys.exit(1)

    if not converter.exists():
        click.echo(f"Error: convert_hf_to_gguf.py not found at {converter}")
        sys.exit(1)
    if not quantizer.exists():
        click.echo(f"Error: llama-quantize not found. Tried multiple paths in {llama_cpp_dir}")
        sys.exit(1)

    total_steps = 3 + (1 if (imatrix and not imatrix_file) else 0)
    step = 0

    # Step 1: Rotate
    step += 1
    click.echo("=" * 60)
    click.echo(f"STEP {step}/{total_steps}: Rotating model weights")
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
    step += 1
    click.echo()
    click.echo("=" * 60)
    click.echo(f"STEP {step}/{total_steps}: Converting to GGUF (F16)")
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

    # Step 2.5: Generate importance matrix (optional)
    resolved_imatrix_file = imatrix_file
    if imatrix and not imatrix_file:
        step += 1
        click.echo()
        click.echo("=" * 60)
        click.echo(f"STEP {step}/{total_steps}: Generating importance matrix")
        click.echo("=" * 60)

        resolved_imatrix_file = str(Path(rotated_dir) / "importance.imatrix")

        cmd_imatrix = [
            str(imatrix_bin),
            "-m", f16_gguf,
            "-o", resolved_imatrix_file,
            "-ngl", "0",  # CPU mode
        ]
        if imatrix_dataset:
            cmd_imatrix.extend(["-f", imatrix_dataset])
            click.echo(f"Calibration dataset: {imatrix_dataset}")
        else:
            click.echo("Using built-in calibration data (for best results, provide --imatrix-dataset)")

        click.echo("This may take 5-15 minutes depending on model size...")
        result = subprocess.run(cmd_imatrix, capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            click.echo(f"Warning: imatrix generation failed:\n{result.stderr}")
            click.echo("Falling back to quantization without importance matrix.")
            resolved_imatrix_file = None
        else:
            click.echo(f"Importance matrix saved to: {resolved_imatrix_file}")

    # Step 3: Quantize
    step += 1
    click.echo()
    click.echo("=" * 60)
    imatrix_note = " + imatrix" if resolved_imatrix_file else ""
    click.echo(f"STEP {step}/{total_steps}: Quantizing to {quant}{imatrix_note}")
    click.echo("=" * 60)

    cmd_quant = [str(quantizer)]
    if resolved_imatrix_file:
        cmd_quant.extend(["--imatrix", resolved_imatrix_file])
    cmd_quant.extend([f16_gguf, output, quant])
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
    click.echo(f"Quant: {quant} with TurboGGUF rotation (seed={seed}){imatrix_note}")
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


@cli.command("kv-compress")
@click.option("--head-dim", "-d", default=128, help="Attention head dimension")
@click.option("--k-bits", default=3, type=click.IntRange(2, 8), help="Bit-width for K cache (default: 3)")
@click.option("--v-bits", default=3, type=click.IntRange(2, 8), help="Bit-width for V cache (default: 3)")
@click.option("--seq-len", "-s", default=4096, help="Sequence length to estimate for")
@click.option("--num-layers", "-l", default=32, help="Number of transformer layers")
@click.option("--num-heads", "-n", default=32, help="Number of attention heads")
@click.option("--model", "-m", help="HuggingFace model ID (auto-detect head-dim/layers/heads)")
def kv_compress(head_dim, k_bits, v_bits, seq_len, num_layers, num_heads, model):
    """Show KV cache compression stats using TurboQuant+.

    Computes memory savings from compressing the KV cache with
    PolarQuant + QJL (TurboQuant, ICLR 2026). Use alongside TurboGGUF
    weight rotation for maximum compression.

    \b
    Usage with llama.cpp:
      --cache-type-k turbo3 --cache-type-v turbo3
    """
    if model:
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model)
            head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            num_layers = config.num_hidden_layers
            num_heads = config.num_attention_heads
            click.echo(f"Model: {model}")
            click.echo(f"  Layers: {num_layers}, Heads: {num_heads}, Head dim: {head_dim}")
            click.echo()
        except Exception as e:
            click.echo(f"Warning: could not load model config ({e}), using provided values")

    from turbogguf.turboquant_plus.kv_cache import KVCacheCompressor

    compressor = KVCacheCompressor(head_dim=head_dim, k_bits=k_bits, v_bits=v_bits)
    stats = compressor.memory_stats(seq_len=seq_len, num_layers=num_layers, num_heads=num_heads)

    click.echo("KV Cache Compression (TurboQuant+)")
    click.echo("=" * 50)
    click.echo(f"Config: K={k_bits}-bit (TurboQuant), V={v_bits}-bit (PolarQuant MSE)")
    click.echo(f"Head dim: {head_dim}, Layers: {num_layers}, Heads: {num_heads}")
    click.echo(f"Sequence length: {seq_len:,}")
    click.echo()
    click.echo(f"Original KV cache (FP16):  {stats['original_mb']:,.1f} MB")
    click.echo(f"Compressed KV cache:       {stats['compressed_mb']:,.1f} MB")
    click.echo(f"Compression ratio:         {stats['compression_ratio']:.1f}x")
    click.echo(f"Memory saved:              {stats['original_mb'] - stats['compressed_mb']:,.1f} MB")
    click.echo()

    # Show recommendations
    click.echo("Recommended llama.cpp flags:")
    cache_type_k = f"turbo{k_bits}" if k_bits <= 4 else f"q{k_bits}_0"
    cache_type_v = f"turbo{v_bits}" if v_bits <= 4 else f"q{v_bits}_0"
    click.echo(f"  --cache-type-k {cache_type_k} --cache-type-v {cache_type_v}")
    click.echo()
    click.echo("Combine with TurboGGUF weight rotation for maximum savings:")
    click.echo("  turbogguf pipeline --model <MODEL> --quant Q2_K --output model.gguf \\")
    click.echo("    --llama-cpp /path/to/llama.cpp")


if __name__ == "__main__":
    cli()
