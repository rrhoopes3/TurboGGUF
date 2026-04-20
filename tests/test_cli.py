"""Tests for llama.cpp tool discovery in the CLI."""

import sys
from pathlib import Path

import pytest

from turbogguf.cli import _find_llama_cpp_tools


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific executable naming")
def test_find_llama_cpp_tools_same_root(tmp_path):
    root = tmp_path / "llama.cpp"
    converter = _touch(root / "convert_hf_to_gguf.py")
    quantizer = _touch(root / "bin" / "llama-quantize.exe")

    found_converter, found_quantizer = _find_llama_cpp_tools(root)

    assert found_converter == converter
    assert found_quantizer == quantizer


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific executable naming")
def test_find_llama_cpp_tools_repo_root(tmp_path):
    root = tmp_path / "llama.cpp"
    converter = _touch(root / "repo" / "convert_hf_to_gguf.py")
    quantizer = _touch(root / "repo" / "bin" / "llama-quantize.exe")

    found_converter, found_quantizer = _find_llama_cpp_tools(root)

    assert found_converter == converter
    assert found_quantizer == quantizer


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific executable naming")
def test_find_llama_cpp_tools_rejects_mixed_source_and_binary_trees(tmp_path):
    root = tmp_path / "llama.cpp"
    _touch(root / "repo" / "convert_hf_to_gguf.py")
    _touch(root / "bin" / "llama-quantize.exe")

    with pytest.raises(FileNotFoundError, match="Refusing to mix converter and binaries"):
        _find_llama_cpp_tools(root)
