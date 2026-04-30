"""TurboGGUF CLI: rotate, quantize, and evaluate LLM models.

Usage:
    turbogguf rotate --model <HF_ID> --output <DIR> [--seed 42]
    turbogguf pipeline --model <HF_ID> --quant Q2_K --output <GGUF> --llama-cpp <DIR>
    turbogguf evaluate --gguf <FILE> --llama-perplexity <BIN>
    turbogguf info --model <HF_ID>
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import click


def _equivalence_gate_options(f):
    """Shared decorator for the forward-equivalence gate flags."""
    f = click.option(
        "--no-equivalence-gate",
        is_flag=True,
        help="Skip the pre/post forward-equivalence check (not recommended).",
    )(f)
    f = click.option(
        "--strict",
        is_flag=True,
        help="Fail with non-zero exit if the equivalence gate doesn't pass.",
    )(f)
    f = click.option(
        "--equivalence-threshold",
        type=float,
        default=1e-4,
        show_default=True,
        help="Max acceptable |logit_rotated - logit_original| on calibration prompts.",
    )(f)
    f = click.option(
        "--calibration-file",
        type=click.Path(exists=True, dir_okay=False),
        default=None,
        help="Optional file with one calibration prompt per line.",
    )(f)
    f = click.option(
        "--calibration-text",
        default=None,
        help="Inline single-prompt override for the equivalence gate.",
    )(f)
    f = click.option(
        "--rotation-precision",
        type=click.Choice(["fp32", "original"]),
        default="fp32",
        show_default=True,
        help="fp32: upcast all weights to fp32 for rotation (recommended). "
        "original: keep legacy per-helper round-trip behavior.",
    )(f)
    return f


def _run_equivalence_gate(
    model,
    tokenizer,
    references,
    *,
    prompts,
    threshold: float,
    strict: bool,
    output_path: Path | None,
):
    """Re-run prompts and emit report/manifest entries. Returns the report dict."""
    from turbogguf.equivalence import compare_logits, EquivalenceFailure

    report = compare_logits(
        model,
        tokenizer,
        prompts=prompts,
        references=references,
        threshold_max_abs=threshold,
        threshold_mean_abs=threshold * 0.5,
    )
    click.echo(report.summary())

    if output_path is not None:
        report_path = output_path / "equivalence_report.json"
        report.write_json(report_path)
        click.echo(f"Equivalence report written to {report_path}")

    if not report.passed:
        if strict:
            raise EquivalenceFailure(report)
        click.echo(
            "Warning: equivalence gate did NOT pass. Re-run with --strict to abort, "
            "or inspect equivalence_report.json for per-prompt stats."
        )
    return report


def _upcast_for_gate(model, rotation_precision: str) -> dict:
    """Upcast model to fp32 when rotation_precision=='fp32' and model isn't already fp32.

    Returns the original dtype map (non-empty means caller must restore after gate).
    When the model is already fp32 or rotation_precision!='fp32', returns {}.

    By upcasting before reference capture and restoring only after the gate runs,
    both the pre- and post-rotation measurements are taken in fp32.  This keeps
    max_abs_diff well inside 1e-4 rather than reflecting fp16 quantisation noise.
    """
    import torch
    from turbogguf.rotation import _collect_param_dtypes, _cast_all_params

    sample = next(model.parameters())
    if rotation_precision != "fp32" or sample.dtype == torch.float32:
        return {}

    original_dtypes = _collect_param_dtypes(model)
    n = _cast_all_params(model, torch.float32)
    click.echo(
        f"Upcasting {n} param(s) to fp32 for reference capture + gate "
        "(both measurements in fp32 keeps max_abs_diff < 1e-4)..."
    )
    return original_dtypes


def _downcast_after_gate(model, original_dtypes: dict, storage_dtype_str: str) -> None:
    """Restore the dtypes saved by _upcast_for_gate."""
    if not original_dtypes:
        return
    from turbogguf.rotation import _restore_param_dtypes
    n = _restore_param_dtypes(model, original_dtypes)
    click.echo(f"Downcasting {n} param(s) back to {storage_dtype_str} for save...")


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """TurboGGUF: Rotation-aware GGUF quantizer.

    Q2 quality that performs like Q4. No llama.cpp patching needed.
    """
    pass


def _parse_max_memory(max_memory_str):
    """Parse --max-memory JSON string into dict.

    Examples:
        '{"cpu": "50GB", "cuda:0": "22GB"}'
        '{"cpu": "55GB"}'
    """
    if not max_memory_str:
        return None
    import json
    raw = json.loads(max_memory_str)
    # accelerate expects integer keys for GPUs, not strings like "0" or "cuda:0"
    parsed = {}
    for k, v in raw.items():
        if k == "cpu" or k == "disk":
            parsed[k] = v
        else:
            # "0", "cuda:0", "1", "cuda:1" etc. -> int
            parsed[int(k.replace("cuda:", ""))] = v
    return parsed


def _llama_exe_name(stem: str) -> str:
    return f"{stem}.exe" if sys.platform == "win32" else stem


def _find_llama_cpp_tool(llama_cpp: str | Path, stem: str) -> Path:
    """Find a llama.cpp executable in the same layouts accepted by the CLI."""
    llama_cpp_dir = Path(llama_cpp)
    exe = _llama_exe_name(stem)
    candidates = [
        llama_cpp_dir / "build" / "bin" / exe,
        llama_cpp_dir / "bin" / exe,
        llama_cpp_dir / exe,
        llama_cpp_dir / "build" / exe,
        llama_cpp_dir / "repo" / "build" / "bin" / exe,
        llama_cpp_dir / "repo" / "bin" / exe,
        llama_cpp_dir / "repo" / exe,
        llama_cpp_dir / "repo" / "build" / exe,
    ]
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        searched = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"{exe} not found under {llama_cpp_dir}.\nSearched: {searched}")
    return found


def _find_llama_cpp_tools(llama_cpp: str | Path) -> tuple[Path, Path]:
    """Find a matched converter + quantizer from one llama.cpp tree.

    Mixing a newer `convert_hf_to_gguf.py` from a source checkout with older
    binaries from a different build can silently produce broken GGUF files.
    Gemma 4 is especially sensitive because the tensor schema is still moving.
    This helper keeps the converter and quantizer on the same tree.
    """
    llama_cpp_dir = Path(llama_cpp)

    converter = None
    for candidate in [
        llama_cpp_dir / "convert_hf_to_gguf.py",
        llama_cpp_dir / "repo" / "convert_hf_to_gguf.py",
    ]:
        if candidate.exists():
            converter = candidate
            break

    if converter is None:
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found in {llama_cpp_dir}"
        )

    tool_root = converter.parent
    exe = _llama_exe_name("llama-quantize")
    quantizer_candidates = [
        tool_root / "build" / "bin" / exe,
        tool_root / "bin" / exe,
        tool_root / exe,
        tool_root / "build" / exe,
    ]

    quantizer = next((p for p in quantizer_candidates if p.exists()), None)
    if quantizer is None and tool_root != llama_cpp_dir:
        # Common dev layout: source checkout in repo/, pre-built binaries
        # alongside it in bin/.  We warn (in case the binaries are from a
        # different release than the source) but don't refuse.
        parent_candidates = [
            llama_cpp_dir / "bin" / exe,
            llama_cpp_dir / "build" / "bin" / exe,
            llama_cpp_dir / exe,
        ]
        parent_quantizer = next((p for p in parent_candidates if p.exists()), None)
        if parent_quantizer is not None:
            click.echo(
                f"Note: converter is under {tool_root} but quantizer is under "
                f"{parent_quantizer.parent}.  If the binaries are from a different "
                "llama.cpp release than the source checkout, conversion may produce "
                "GGUFs the quantizer can't load."
            )
            quantizer = parent_quantizer

    if quantizer is None:
        searched = ", ".join(str(p) for p in quantizer_candidates)
        raise FileNotFoundError(
            f"{exe} not found under {llama_cpp_dir}.\nSearched: {searched}"
        )

    return converter, quantizer


def _run_checked(cmd: list[str], *, label: str, timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run a subprocess and raise with useful stderr/stdout context on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        details = result.stderr or result.stdout
        raise RuntimeError(f"{label} failed:\n{details}")
    return result


def _convert_hf_to_f16_gguf(converter: Path, hf_dir: str | Path, gguf_path: str | Path) -> None:
    cmd = [
        sys.executable, str(converter),
        str(hf_dir),
        "--outtype", "f16",
        "--outfile", str(gguf_path),
    ]
    _run_checked(cmd, label="GGUF conversion")

    from turbogguf.export import patch_gguf_output_tensor
    patched = patch_gguf_output_tensor(str(gguf_path))
    if patched:
        click.echo("  (output tensor renamed for pre-built llama.cpp compatibility)")


def _run_imatrix(
    imatrix_bin: Path,
    f16_gguf: str | Path,
    text_path: str | Path,
    output_path: str | Path,
    *,
    chunks: int,
    context_size: int,
    n_gpu_layers: int = 0,
) -> None:
    cmd = [
        str(imatrix_bin),
        "-m", str(f16_gguf),
        "-f", str(text_path),
        "-o", str(output_path),
        "-c", str(context_size),
        "--chunks", str(chunks),
        "-ngl", str(n_gpu_layers),
    ]
    _run_checked(cmd, label="llama-imatrix", timeout=3600)


def _quantize_gguf(
    quantizer: Path,
    f16_gguf: str | Path,
    output_path: str | Path,
    quant: str,
    *,
    imatrix_path: str | Path | None = None,
) -> None:
    cmd = [str(quantizer)]
    if imatrix_path is not None:
        cmd.extend(["--imatrix", str(imatrix_path)])
    cmd.extend([str(f16_gguf), str(output_path), quant])
    _run_checked(cmd, label="Quantization")


def _evaluate_gguf_ppl(
    perplexity_bin: Path,
    gguf_path: str | Path,
    text_path: str | Path,
    *,
    chunks: int,
    context_size: int,
    label: str,
    n_gpu_layers: int = 0,
):
    from turbogguf.evaluate import evaluate_gguf

    return evaluate_gguf(
        str(gguf_path),
        str(perplexity_bin),
        dataset=str(text_path),
        context_size=context_size,
        label=label,
        chunks=chunks,
        n_gpu_layers=n_gpu_layers,
    )


def _default_auto_text_path() -> Path:
    return Path(__file__).parent / "data" / "auto_calibration.txt"


def _write_auto_report(
    output: str | Path,
    *,
    stock_result,
    rotated_result,
    verdict: str,
    margin: float,
    chunks: int,
    context_size: int,
    text_path: str | Path,
    quant: str,
    imatrix: bool,
) -> Path:
    report_path = Path(output).with_name("comparison_report.json")
    data = {
        "quant": quant,
        "chunks": chunks,
        "context_size": context_size,
        "text_path": str(text_path),
        "imatrix": imatrix,
        "win_margin_ppl": margin,
        "stock": stock_result.__dict__,
        "rotated": rotated_result.__dict__,
        "verdict": verdict,
    }
    report_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return report_path


@cli.command()
@click.option("--model", "-m", required=True, help="HuggingFace model ID or local path")
@click.option("--output", "-o", required=True, help="Output directory for rotated model")
@click.option("--seed", default=42, help="Random seed for Hadamard rotation")
@click.option("--no-r2", is_flag=True, help="Skip per-head R2 rotation")
@click.option("--dtype", default="float16", type=click.Choice(["float16", "bfloat16"]))
@click.option("--device-map", default="cpu", help="Device map: 'cpu', 'auto', or 'cuda:0'")
@click.option("--max-memory", default=None, help='Max memory JSON, e.g. \'{"cpu": "50GB", "cuda:0": "22GB"}\'')
@click.option("--trust-remote-code", is_flag=True, help="Trust remote code for model loading")
@click.option(
    "--audit-only",
    is_flag=True,
    help="Run rotation + equivalence gate, print the report, but do NOT save the rotated model.",
)
@_equivalence_gate_options
def rotate(
    model,
    output,
    seed,
    no_r2,
    dtype,
    device_map,
    max_memory,
    trust_remote_code,
    audit_only,
    rotation_precision,
    calibration_text,
    calibration_file,
    equivalence_threshold,
    strict,
    no_equivalence_gate,
):
    """Apply Hadamard rotation to model weights.

    Loads a HuggingFace model, captures reference logits on a small set of
    calibration prompts, fuses RMSNorm weights, applies R1+R2 rotations,
    re-runs the prompts and verifies the logits match, then saves as a
    standard HF checkpoint. The rotated model produces near-identical
    outputs but quantizes much better at low bit widths (Q2_K, Q3_K).
    """
    import torch
    from turbogguf.model_loader import load_model
    from turbogguf.rotation import rotate_model
    from turbogguf.export import export_rotated_model
    from turbogguf.equivalence import (
        capture_logits,
        load_prompts,
        EquivalenceFailure,
    )

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    mem = _parse_max_memory(max_memory)

    click.echo(f"TurboGGUF v0.1.0 — Rotating {model}")
    click.echo(
        f"Seed: {seed}, R2: {'disabled' if no_r2 else 'enabled'}, "
        f"rotation precision: {rotation_precision}"
    )
    if mem:
        click.echo(f"Memory map: {mem}")
    click.echo()

    model_obj, tokenizer, handler = load_model(
        model,
        dtype=dtype_map[dtype],
        device_map=device_map,
        max_memory=mem,
        trust_remote_code=trust_remote_code,
    )

    # Record storage dtype now, before any upcast, so metadata reflects save dtype.
    storage_dtype_str = str(next(model_obj.parameters()).dtype).removeprefix("torch.")

    # Upcast to fp32 before reference capture so both measurements are in fp32.
    # rotate_model will see the model already in fp32 and skip its internal upcast.
    cli_original_dtypes = {}
    if not no_equivalence_gate:
        cli_original_dtypes = _upcast_for_gate(model_obj, rotation_precision)

    references = None
    prompts: list[str] = []
    if not no_equivalence_gate:
        prompts = load_prompts(
            calibration_file=calibration_file,
            calibration_text=calibration_text,
        )
        click.echo(f"Capturing reference logits on {len(prompts)} calibration prompt(s)...")
        references = capture_logits(model_obj, tokenizer, prompts)

    metadata = rotate_model(
        model_obj,
        handler=handler,
        seed=seed,
        apply_r2=not no_r2,
        rotation_precision=rotation_precision,
    )
    if cli_original_dtypes:
        metadata["storage_dtype"] = storage_dtype_str

    output_path = Path(output)
    if not audit_only:
        output_path.mkdir(parents=True, exist_ok=True)

    report = None
    try:
        if not no_equivalence_gate:
            report = _run_equivalence_gate(
                model_obj,
                tokenizer,
                references,
                prompts=prompts,
                threshold=equivalence_threshold,
                strict=strict,
                output_path=output_path if not audit_only else None,
            )
            metadata["equivalence"] = report.to_dict()
        else:
            click.echo("Equivalence gate skipped (--no-equivalence-gate).")
    except EquivalenceFailure as e:
        _downcast_after_gate(model_obj, cli_original_dtypes, storage_dtype_str)
        if not audit_only:
            metadata["equivalence"] = e.report.to_dict()
            (output_path / "rotation_manifest.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
        click.echo(f"Aborting (--strict): {e}")
        sys.exit(1)

    # Downcast after gate so the gate ran in fp32, save is in original dtype.
    _downcast_after_gate(model_obj, cli_original_dtypes, storage_dtype_str)

    if audit_only:
        click.echo("Audit-only mode: skipping save.")
        click.echo(json.dumps(metadata, indent=2))
        return

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
@click.option("--device-map", default="cpu", help="Device map: 'cpu', 'auto', or 'cuda:0'")
@click.option("--max-memory", default=None, help='Max memory JSON, e.g. \'{"cpu": "50GB", "cuda:0": "22GB"}\'')
@click.option("--dtype", default="float16", type=click.Choice(["float16", "bfloat16"]),
              help="Load dtype. Use bfloat16 for models stored as bf16 to avoid doubling RAM during conversion.")
@click.option("--trust-remote-code", is_flag=True, help="Trust remote code")
@click.option("--keep-intermediate", is_flag=True, help="Keep intermediate files")
@click.option("--auto", is_flag=True, help="Quantize stock and rotated candidates, evaluate both, and keep the better GGUF.")
@click.option("--auto-text", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Text file used for --auto imatrix + perplexity. Defaults to bundled calibration text.")
@click.option("--auto-chunks", default=20, show_default=True, type=click.IntRange(1),
              help="Number of chunks for --auto imatrix/perplexity bake-off.")
@click.option("--auto-context-size", default=512, show_default=True, type=click.IntRange(128),
              help="Context size for --auto imatrix/perplexity bake-off.")
@click.option("--auto-margin", default=0.01, show_default=True, type=float,
              help="Rotated candidate must beat stock by at least this PPL delta.")
@click.option("--no-auto-imatrix", is_flag=True,
              help="Disable imatrix during --auto quantization.")
@click.option("--auto-ngl", default=0, show_default=True, type=click.IntRange(0),
              help="GPU layers for --auto imatrix/perplexity (0=CPU, 99=all). Set to 99 to use the GPU and finish in minutes instead of hours.")
@click.option("--auto-workdir", type=click.Path(file_okay=False), default=None,
              help="Directory for --auto intermediate files (stock + rotated HF dirs, F16 GGUFs, imatrix). Defaults to <output_parent>/auto_intermediate so big files land next to your final GGUF, not on the system temp drive.")
@_equivalence_gate_options
def pipeline(
    model,
    output,
    quant,
    seed,
    llama_cpp,
    no_r2,
    device_map,
    max_memory,
    dtype,
    trust_remote_code,
    keep_intermediate,
    auto,
    auto_text,
    auto_chunks,
    auto_context_size,
    auto_margin,
    no_auto_imatrix,
    auto_ngl,
    auto_workdir,
    rotation_precision,
    calibration_text,
    calibration_file,
    equivalence_threshold,
    strict,
    no_equivalence_gate,
):
    """Full pipeline: rotate -> convert to GGUF -> quantize.

    One command to go from HuggingFace model to quantized GGUF.
    """
    import torch
    import tempfile
    import shutil
    import gc
    from turbogguf.model_loader import load_model
    from turbogguf.rotation import rotate_model
    from turbogguf.export import export_rotated_model
    from turbogguf.equivalence import (
        capture_logits,
        load_prompts,
        EquivalenceFailure,
    )

    mem = _parse_max_memory(max_memory)

    try:
        converter, quantizer = _find_llama_cpp_tools(llama_cpp)
        perplexity_bin = _find_llama_cpp_tool(llama_cpp, "llama-perplexity") if auto else None
        imatrix_bin = None
        if auto and not no_auto_imatrix:
            imatrix_bin = _find_llama_cpp_tool(llama_cpp, "llama-imatrix")
    except FileNotFoundError as e:
        click.echo(f"Error: {e}")
        sys.exit(1)

    click.echo(f"Using converter: {converter}")
    click.echo(f"Using quantizer: {quantizer}")
    if auto:
        click.echo(f"Using perplexity: {perplexity_bin}")
        if imatrix_bin is not None:
            click.echo(f"Using imatrix: {imatrix_bin}")

    # Step 1: Rotate
    click.echo("=" * 60)
    click.echo("STEP 1/3: Preparing model weights")
    click.echo("=" * 60)
    if mem:
        click.echo(f"Memory map: {mem}")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    model_obj, tokenizer, handler = load_model(
        model, dtype=dtype_map[dtype], device_map=device_map, max_memory=mem,
        trust_remote_code=trust_remote_code,
    )

    storage_dtype_str = str(next(model_obj.parameters()).dtype).removeprefix("torch.")
    auto_text_path = Path(auto_text) if auto_text else _default_auto_text_path()
    work_dir = None
    stock_dir = None

    if auto:
        if auto_workdir:
            work_dir = Path(auto_workdir)
        else:
            # Default next to the output GGUF so the ~5x output-size of
            # intermediate files lands on the same drive as --output rather
            # than the system temp drive (which is often much smaller).
            work_dir = Path(output).parent / "auto_intermediate"
        work_dir.mkdir(parents=True, exist_ok=True)

        stock_dir = work_dir / "stock_hf"
        click.echo(f"Saving stock candidate to {stock_dir}...")
        stock_dir.mkdir(parents=True, exist_ok=True)
        model_obj.save_pretrained(
            str(stock_dir),
            safe_serialization=True,
            max_shard_size="4GB",
        )
        tokenizer.save_pretrained(str(stock_dir))

    cli_original_dtypes = {}
    if not no_equivalence_gate:
        cli_original_dtypes = _upcast_for_gate(model_obj, rotation_precision)

    references = None
    prompts: list[str] = []
    if not no_equivalence_gate:
        prompts = load_prompts(
            calibration_file=calibration_file,
            calibration_text=calibration_text,
        )
        click.echo(f"Capturing reference logits on {len(prompts)} calibration prompt(s)...")
        references = capture_logits(model_obj, tokenizer, prompts)

    metadata = rotate_model(
        model_obj,
        handler=handler,
        seed=seed,
        apply_r2=not no_r2,
        rotation_precision=rotation_precision,
    )
    if cli_original_dtypes:
        metadata["storage_dtype"] = storage_dtype_str

    if not no_equivalence_gate:
        try:
            report = _run_equivalence_gate(
                model_obj,
                tokenizer,
                references,
                prompts=prompts,
                threshold=equivalence_threshold,
                strict=strict,
                output_path=None,  # written into the rotated dir below
            )
            metadata["equivalence"] = report.to_dict()
        except EquivalenceFailure as e:
            _downcast_after_gate(model_obj, cli_original_dtypes, storage_dtype_str)
            metadata["equivalence"] = e.report.to_dict()
            click.echo(f"Aborting (--strict): {e}")
            sys.exit(1)

    # Downcast after gate so gate ran in fp32; model saved in original dtype.
    _downcast_after_gate(model_obj, cli_original_dtypes, storage_dtype_str)

    if keep_intermediate:
        rotated_parent = work_dir if auto and work_dir is not None else Path(output).parent
        rotated_dir = str(rotated_parent / "rotated_intermediate")
    else:
        rotated_dir = str(work_dir / "rotated_hf") if auto and work_dir is not None else tempfile.mkdtemp(prefix="turbogguf_")

    export_rotated_model(model_obj, tokenizer, rotated_dir, metadata=metadata)

    # Free model memory
    del model_obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 2: Convert to GGUF
    click.echo()
    click.echo("=" * 60)
    click.echo("STEP 2/3: Converting to GGUF (F16)")
    click.echo("=" * 60)

    f16_gguf = str(Path(rotated_dir) / "model.gguf")
    try:
        _convert_hf_to_f16_gguf(converter, rotated_dir, f16_gguf)
        stock_f16_gguf = None
        if auto:
            stock_f16_gguf = work_dir / "stock-f16.gguf"
            click.echo("Converting stock candidate to GGUF (F16)...")
            _convert_hf_to_f16_gguf(converter, stock_dir, stock_f16_gguf)
    except RuntimeError as e:
        click.echo(str(e))
        sys.exit(1)
    click.echo("GGUF conversion complete.")

    # Pre-built llama.cpp ≤b8638 looks for tensor "output" (no .weight suffix)
    # Step 3: Quantize
    click.echo()
    click.echo("=" * 60)
    click.echo(f"STEP 3/3: Quantizing to {quant}")
    click.echo("=" * 60)

    try:
        if auto:
            rotated_quant_gguf = work_dir / "rotated-quant.gguf"
            stock_quant_gguf = work_dir / "stock-quant.gguf"
            rotated_imatrix = None
            stock_imatrix = None

            if imatrix_bin is not None:
                rotated_imatrix = work_dir / "rotated.imatrix"
                stock_imatrix = work_dir / "stock.imatrix"
                click.echo(f"Building stock imatrix on {auto_chunks} chunk(s) (ngl={auto_ngl})...")
                _run_imatrix(
                    imatrix_bin, stock_f16_gguf, auto_text_path, stock_imatrix,
                    chunks=auto_chunks, context_size=auto_context_size,
                    n_gpu_layers=auto_ngl,
                )
                click.echo(f"Building rotated imatrix on {auto_chunks} chunk(s) (ngl={auto_ngl})...")
                _run_imatrix(
                    imatrix_bin, f16_gguf, auto_text_path, rotated_imatrix,
                    chunks=auto_chunks, context_size=auto_context_size,
                    n_gpu_layers=auto_ngl,
                )

            click.echo("Quantizing stock candidate...")
            _quantize_gguf(
                quantizer, stock_f16_gguf, stock_quant_gguf, quant,
                imatrix_path=stock_imatrix,
            )
            click.echo("Quantizing rotated candidate...")
            _quantize_gguf(
                quantizer, f16_gguf, rotated_quant_gguf, quant,
                imatrix_path=rotated_imatrix,
            )

            click.echo()
            click.echo("=" * 60)
            click.echo(f"AUTO: Evaluating {auto_chunks} chunk(s) (ngl={auto_ngl})")
            click.echo("=" * 60)
            stock_result = _evaluate_gguf_ppl(
                perplexity_bin, stock_quant_gguf, auto_text_path,
                chunks=auto_chunks, context_size=auto_context_size, label="stock",
                n_gpu_layers=auto_ngl,
            )
            rotated_result = _evaluate_gguf_ppl(
                perplexity_bin, rotated_quant_gguf, auto_text_path,
                chunks=auto_chunks, context_size=auto_context_size, label="rotated",
                n_gpu_layers=auto_ngl,
            )

            rotated_wins = rotated_result.perplexity <= stock_result.perplexity - auto_margin
            winner = rotated_quant_gguf if rotated_wins else stock_quant_gguf
            verdict = "rotated" if rotated_wins else "stock"

            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists():
                output_path.unlink()
            shutil.copy2(winner, output_path)

            report_path = _write_auto_report(
                output,
                stock_result=stock_result,
                rotated_result=rotated_result,
                verdict=verdict,
                margin=auto_margin,
                chunks=auto_chunks,
                context_size=auto_context_size,
                text_path=auto_text_path,
                quant=quant,
                imatrix=imatrix_bin is not None,
            )

            click.echo(f"Stock PPL:   {stock_result.perplexity:.4f}")
            click.echo(f"Rotated PPL: {rotated_result.perplexity:.4f}")
            if rotated_wins:
                delta = stock_result.perplexity - rotated_result.perplexity
                click.echo(f"Verdict: rotated won by {delta:.4f} PPL; keeping rotated.")
            else:
                delta = rotated_result.perplexity - stock_result.perplexity
                click.echo(f"Verdict: stock was better/within margin ({delta:.4f} PPL); keeping stock.")
            click.echo(f"Comparison report written to {report_path}")
        else:
            _quantize_gguf(quantizer, f16_gguf, output, quant)
    except RuntimeError as e:
        click.echo(str(e))
        sys.exit(1)

    # Cleanup
    if not keep_intermediate and auto and work_dir is not None:
        shutil.rmtree(work_dir, ignore_errors=True)
    elif not keep_intermediate:
        shutil.rmtree(rotated_dir, ignore_errors=True)

    output_size = Path(output).stat().st_size / (1024**3)
    click.echo()
    click.echo("=" * 60)
    click.echo(f"Done! Output: {output} ({output_size:.2f} GB)")
    if auto:
        click.echo(f"Quant: {quant} with --auto bake-off (seed={seed})")
    else:
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
