"""Forward-equivalence gate tests.

These tests do not require any HuggingFace download. They exercise both the
PASS path (full rotation preserves logits to the threshold) and the FAIL
path (a deliberately corrupted "rotation" produces a divergence the gate
catches), plus the upcast/downcast helpers and the prompt loader.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tests.test_roundtrip import MiniTransformer, rotate_mini_transformer
from turbogguf import equivalence
from turbogguf.equivalence import (
    DEFAULT_THRESHOLD_MAX_ABS,
    EquivalenceFailure,
    capture_logits,
    compare_logits,
    load_default_prompts,
    load_prompts,
    parse_prompts_text,
)
from turbogguf.rotation import (
    _cast_all_params,
    _collect_param_dtypes,
    _restore_param_dtypes,
)


class DummyTokenizer:
    """Hash-based tokenizer keyed to a synthetic vocab (no external deps).

    Returns the dict shape that capture_logits / compare_logits expect: a
    transformers-style {"input_ids": LongTensor[1, T]}.
    """

    def __init__(self, vocab: int):
        self.vocab = vocab

    def __call__(self, prompt, return_tensors="pt", truncation=True,
                 max_length=64, add_special_tokens=True):
        # Deterministic per-character ids; clamp to vocab size.
        ids = [(ord(c) % (self.vocab - 1)) + 1 for c in (prompt or " ")]
        ids = ids[:max_length] or [1]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class DummyOutput:
    def __init__(self, logits):
        self.logits = logits


class WrappedMiniTransformer(nn.Module):
    """MiniTransformer wrapper that returns a transformers-style output object."""

    def __init__(self, inner: MiniTransformer):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids=None):
        return DummyOutput(self.inner(input_ids))


@pytest.fixture
def synth_model():
    torch.manual_seed(0)
    inner = MiniTransformer(vocab=64, hidden=32, num_heads=4, num_layers=2,
                            intermediate=64)
    inner.eval()
    return WrappedMiniTransformer(inner), DummyTokenizer(vocab=64)


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------


def test_default_prompts_nonempty():
    prompts = load_default_prompts()
    assert len(prompts) >= 5, "ship at least a handful of diverse prompts"
    assert all(isinstance(p, str) and p for p in prompts)


def test_parse_prompts_strips_comments_and_blanks():
    text = "# header\n\nfirst\n  \n# mid\nsecond\n"
    assert parse_prompts_text(text) == ["first", "second"]


def test_parse_prompts_falls_back_to_single_prompt():
    assert parse_prompts_text("only this one") == ["only this one"]


def test_load_prompts_priority_text_over_file(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("from-file\n", encoding="utf-8")
    assert load_prompts(calibration_file=str(f),
                        calibration_text="from-text") == ["from-text"]
    assert load_prompts(calibration_file=str(f)) == ["from-file"]


# ---------------------------------------------------------------------------
# Upcast / downcast helpers
# ---------------------------------------------------------------------------


def test_cast_all_params_round_trip():
    torch.manual_seed(0)
    model = MiniTransformer(vocab=32, hidden=16, num_heads=2, num_layers=1,
                            intermediate=32).to(torch.float16)

    original = _collect_param_dtypes(model)
    assert all(dt == torch.float16 for dt in original.values())

    n = _cast_all_params(model, torch.float32)
    assert n > 0
    assert all(p.dtype == torch.float32 for p in model.parameters())

    restored = _restore_param_dtypes(model, original)
    assert restored == n
    assert all(p.dtype == torch.float16 for p in model.parameters())


# ---------------------------------------------------------------------------
# PASS path: full rotation preserves logits within threshold
# ---------------------------------------------------------------------------


def test_gate_passes_after_correct_rotation(synth_model):
    model, tok = synth_model
    prompts = ["hello world", "the quick brown fox", "1 + 1 ="]

    refs = capture_logits(model, tok, prompts)
    rotate_mini_transformer(model.inner, seed=42)
    report = compare_logits(model, tok, prompts, refs,
                            threshold_max_abs=DEFAULT_THRESHOLD_MAX_ABS)

    assert report.passed, report.summary()
    assert report.max_abs_diff <= DEFAULT_THRESHOLD_MAX_ABS
    assert len(report.per_prompt) == len(prompts)
    assert all(s.seq_len > 0 for s in report.per_prompt)


# ---------------------------------------------------------------------------
# FAIL path: a corrupted rotation must be caught
# ---------------------------------------------------------------------------


def _corrupt_rotation(model: MiniTransformer):
    """Apply a real rotation, then perturb one weight enough to fail the gate.

    This simulates exactly the failure mode the gate is meant to catch:
    the math says R^T @ R = I, but a buggy step (precision drift, missed
    bias rotation, wrong basis) leaves a residual mismatch in the chain.
    """
    rotate_mini_transformer(model, seed=42)
    with torch.no_grad():
        model.lm_head.weight.data[0, 0] += 0.5


def test_gate_fails_on_corrupted_rotation(synth_model):
    model, tok = synth_model
    prompts = ["hello world", "second prompt"]

    refs = capture_logits(model, tok, prompts)
    _corrupt_rotation(model.inner)
    report = compare_logits(model, tok, prompts, refs,
                            threshold_max_abs=DEFAULT_THRESHOLD_MAX_ABS)

    assert not report.passed, "gate must catch a 0.5-magnitude weight perturbation"
    assert report.max_abs_diff > DEFAULT_THRESHOLD_MAX_ABS


def test_strict_failure_raises_with_report(synth_model):
    model, tok = synth_model
    prompts = ["only one"]

    refs = capture_logits(model, tok, prompts)
    _corrupt_rotation(model.inner)
    report = compare_logits(model, tok, prompts, refs,
                            threshold_max_abs=DEFAULT_THRESHOLD_MAX_ABS)
    assert not report.passed

    err = EquivalenceFailure(report)
    assert err.report is report
    assert "FAIL" in err.report.summary()


def test_report_serializes_to_json(tmp_path, synth_model):
    model, tok = synth_model
    prompts = ["serialize me"]
    refs = capture_logits(model, tok, prompts)
    rotate_mini_transformer(model.inner, seed=42)
    report = compare_logits(model, tok, prompts, refs)

    out = tmp_path / "equivalence_report.json"
    report.write_json(out)
    payload = json.loads(out.read_text(encoding="utf-8"))

    for key in ("max_abs_diff", "mean_abs_diff", "passed", "per_prompt",
                "threshold_max_abs", "storage_dtype"):
        assert key in payload
    assert isinstance(payload["per_prompt"], list)
    assert len(payload["per_prompt"]) == 1
