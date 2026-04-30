"""Forward-equivalence gate for TurboGGUF rotation.

The rotation pipeline is mathematically a no-op on the model's input/output
behavior: at every layer boundary R^T @ R = I, so the rotated weights must
produce the same logits (up to floating-point noise) as the original weights.

In practice, fp16/bf16 round-trips inside the rotation helpers can compound
across norm fusion + R1 + R2 + per-head bias rotation. This module captures
reference logits before the pipeline runs and re-runs the same prompts after,
then reports per-prompt and aggregate divergence statistics. The gate is the
single check that would have caught the LLaMA bf16 drift immediately.

Usage:
    pre = capture_logits(model, tokenizer, prompts)
    rotate_model(model, ...)
    report = compare_logits(model, tokenizer, prompts, pre)
    report.summary()  # one-line PASS/FAIL summary
    report.to_dict()  # for the rotation manifest
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Sequence

import torch


DEFAULT_MAX_SEQ_LEN = 64
DEFAULT_THRESHOLD_MAX_ABS = 1e-4
DEFAULT_THRESHOLD_MEAN_ABS = 5e-5


def _bundled_prompts_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "calibration_prompts.txt"


def load_default_prompts() -> list[str]:
    """Load the bundled diverse calibration prompts."""
    return parse_prompts_text(_bundled_prompts_path().read_text(encoding="utf-8"))


def parse_prompts_text(text: str) -> list[str]:
    """Parse a prompts text blob.

    One prompt per line; blank lines and lines starting with '#' are dropped.
    If the entire blob has no newlines (or every line is a comment) it is
    returned as a single prompt.
    """
    lines = [ln.rstrip("\r\n") for ln in text.splitlines()]
    prompts = [
        ln for ln in lines
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if not prompts:
        stripped = text.strip()
        return [stripped] if stripped else []
    return prompts


def load_prompts(
    calibration_file: Optional[str] = None,
    calibration_text: Optional[str] = None,
) -> list[str]:
    """Resolve prompts in priority order: explicit text > file > bundled defaults."""
    if calibration_text is not None:
        return parse_prompts_text(calibration_text)
    if calibration_file is not None:
        return parse_prompts_text(Path(calibration_file).read_text(encoding="utf-8"))
    return load_default_prompts()


@dataclass
class PerPromptStat:
    prompt_index: int
    seq_len: int
    max_abs_diff: float
    mean_abs_diff: float
    kl_div: float


@dataclass
class EquivalenceReport:
    """Aggregate diagnostics from a forward-equivalence comparison."""

    num_prompts: int
    storage_dtype: str
    threshold_max_abs: float
    threshold_mean_abs: float
    max_abs_diff: float
    mean_abs_diff: float
    max_kl_div: float
    mean_kl_div: float
    passed: bool
    per_prompt: list[PerPromptStat] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: Optional[str] = None

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[equivalence] max_abs_diff={self.max_abs_diff:.3e} "
            f"mean_abs_diff={self.mean_abs_diff:.3e} "
            f"max_kl={self.max_kl_div:.3e} ({status}) "
            f"prompts={self.num_prompts} dtype={self.storage_dtype} "
            f"threshold={self.threshold_max_abs:.0e}"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _tokenize(tokenizer, prompt: str, max_seq_len: int):
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=True,
    )
    return enc["input_ids"]


def _logits_for_prompt(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Forward a single prompt and return last-token logits in fp32 on CPU."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    out = model(input_ids=input_ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    # Keep only the last position to bound memory; that's what generation uses
    # and it's where a quality regression manifests first.
    last = logits[0, -1, :].detach().to("cpu", dtype=torch.float32)
    return last


@torch.no_grad()
def capture_logits(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
) -> list[dict]:
    """Run prompts through the model and stash reference logits.

    Returns a list of {prompt, input_ids, logits} dicts, one per prompt.
    Stored in fp32 on CPU so a later compare_logits() against the rotated
    model is independent of where the model lives.
    """
    refs = []
    was_training = model.training
    model.eval()
    try:
        for i, prompt in enumerate(prompts):
            input_ids = _tokenize(tokenizer, prompt, max_seq_len)
            logits = _logits_for_prompt(model, input_ids)
            refs.append({
                "prompt": prompt,
                "input_ids": input_ids.detach().to("cpu"),
                "logits": logits,
                "index": i,
            })
    finally:
        if was_training:
            model.train()
    return refs


def _kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(P || Q) between two logit vectors, in nats. Returns a finite float."""
    p = torch.log_softmax(p_logits.double(), dim=-1)
    q = torch.log_softmax(q_logits.double(), dim=-1)
    kl = (p.exp() * (p - q)).sum().item()
    if not math.isfinite(kl):
        return float("inf")
    return float(kl)


@torch.no_grad()
def compare_logits(
    model,
    tokenizer,
    prompts: Sequence[str],
    references: list[dict],
    threshold_max_abs: float = DEFAULT_THRESHOLD_MAX_ABS,
    threshold_mean_abs: float = DEFAULT_THRESHOLD_MEAN_ABS,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
) -> EquivalenceReport:
    """Re-run prompts through the (rotated) model and diff against references."""
    if len(references) != len(prompts):
        raise ValueError(
            f"prompt/reference count mismatch: {len(prompts)} vs {len(references)}"
        )

    sample_param = next(model.parameters())
    storage_dtype_str = str(sample_param.dtype).removeprefix("torch.")

    start = time.perf_counter()
    per_prompt: list[PerPromptStat] = []
    max_abs_total = 0.0
    sum_mean_abs = 0.0
    max_kl = 0.0
    sum_kl = 0.0

    was_training = model.training
    model.eval()
    error_msg: Optional[str] = None
    try:
        for i, prompt in enumerate(prompts):
            ref = references[i]
            ref_logits = ref["logits"]
            input_ids = ref["input_ids"]
            new_logits = _logits_for_prompt(model, input_ids)

            diff = (new_logits - ref_logits).abs()
            mx = float(diff.max().item())
            mn = float(diff.mean().item())
            kl = _kl_divergence(ref_logits, new_logits)

            per_prompt.append(PerPromptStat(
                prompt_index=i,
                seq_len=int(input_ids.shape[-1]),
                max_abs_diff=mx,
                mean_abs_diff=mn,
                kl_div=kl,
            ))
            max_abs_total = max(max_abs_total, mx)
            sum_mean_abs += mn
            max_kl = max(max_kl, kl)
            sum_kl += kl
    except Exception as e:  # pragma: no cover - defensive
        error_msg = f"{type(e).__name__}: {e}"
    finally:
        if was_training:
            model.train()

    n = max(len(prompts), 1)
    mean_abs = sum_mean_abs / n
    mean_kl = sum_kl / n

    passed = (
        error_msg is None
        and max_abs_total <= threshold_max_abs
        and mean_abs <= threshold_mean_abs
    )

    return EquivalenceReport(
        num_prompts=len(prompts),
        storage_dtype=storage_dtype_str,
        threshold_max_abs=threshold_max_abs,
        threshold_mean_abs=threshold_mean_abs,
        max_abs_diff=max_abs_total,
        mean_abs_diff=mean_abs,
        max_kl_div=max_kl,
        mean_kl_div=mean_kl,
        passed=passed,
        per_prompt=per_prompt,
        elapsed_seconds=time.perf_counter() - start,
        error=error_msg,
    )


class EquivalenceFailure(RuntimeError):
    """Raised when --strict is set and the equivalence gate fails."""

    def __init__(self, report: EquivalenceReport):
        super().__init__(report.summary())
        self.report = report
