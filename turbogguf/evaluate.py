"""Perplexity evaluation for comparing rotated vs unrotated quantizations.

Uses llama.cpp's llama-perplexity binary for GGUF evaluation, or
HuggingFace's evaluate library for direct model evaluation.
"""

import subprocess
import re
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class PerplexityResult:
    """Result of a perplexity evaluation."""
    model_path: str
    perplexity: float
    tokens: int
    label: str = ""

    def __repr__(self):
        return f"PPL({self.label}: {self.perplexity:.4f}, {self.tokens} tokens)"


def evaluate_gguf(
    gguf_path: str,
    llama_perplexity_bin: str,
    dataset: str = "wikitext-2",
    context_size: int = 2048,
    label: str = "",
) -> PerplexityResult:
    """Evaluate a GGUF model's perplexity using llama-perplexity.

    Args:
        gguf_path: Path to the GGUF file
        llama_perplexity_bin: Path to llama-perplexity binary
        dataset: Dataset to evaluate on
        context_size: Context window size
        label: Human-readable label for this run

    Returns:
        PerplexityResult with measured perplexity
    """
    cmd = [
        llama_perplexity_bin,
        "-m", gguf_path,
        "--perplexity",
        "-c", str(context_size),
        "-ngl", "0",  # CPU mode for consistency
    ]

    print(f"Running perplexity evaluation: {label or gguf_path}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        raise RuntimeError(f"llama-perplexity failed: {result.stderr}")

    # Parse perplexity from output
    # Expected format: "Final estimate: PPL = 6.1234 +/- 0.0123"
    ppl_match = re.search(r"Final estimate: PPL = ([\d.]+)", result.stdout)
    if not ppl_match:
        # Try alternative format
        ppl_match = re.search(r"perplexity = ([\d.]+)", result.stdout)
    if not ppl_match:
        raise ValueError(f"Could not parse perplexity from output:\n{result.stdout[-500:]}")

    ppl = float(ppl_match.group(1))

    tokens_match = re.search(r"(\d+) tokens", result.stdout)
    tokens = int(tokens_match.group(1)) if tokens_match else 0

    return PerplexityResult(
        model_path=gguf_path,
        perplexity=ppl,
        tokens=tokens,
        label=label or Path(gguf_path).stem,
    )


def compare_models(
    results: list[PerplexityResult],
) -> str:
    """Format a comparison table of perplexity results.

    Args:
        results: List of PerplexityResult objects

    Returns:
        Formatted comparison string
    """
    if not results:
        return "No results to compare."

    lines = [
        "=" * 60,
        "PERPLEXITY COMPARISON",
        "=" * 60,
        f"{'Label':<30} {'PPL':>10} {'Tokens':>10}",
        "-" * 60,
    ]

    best = min(results, key=lambda r: r.perplexity)

    for r in sorted(results, key=lambda r: r.perplexity):
        marker = " *BEST*" if r is best else ""
        lines.append(f"{r.label:<30} {r.perplexity:>10.4f} {r.tokens:>10}{marker}")

    lines.append("-" * 60)

    if len(results) >= 2:
        worst = max(results, key=lambda r: r.perplexity)
        improvement = (worst.perplexity - best.perplexity) / worst.perplexity * 100
        lines.append(f"Best improvement: {improvement:.1f}% PPL reduction")

    lines.append("=" * 60)
    return "\n".join(lines)


def save_results(
    results: list[PerplexityResult],
    output_path: str,
) -> None:
    """Save evaluation results to JSON.

    Args:
        results: List of PerplexityResult objects
        output_path: Path to save JSON file
    """
    data = {
        "results": [asdict(r) for r in results],
        "comparison": compare_models(results),
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {output_path}")
