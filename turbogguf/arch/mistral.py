"""Mistral architecture handler.

Mistral uses the same structure as LLaMA with sliding window attention.
"""

from turbogguf.arch.llama import LlamaHandler


class MistralHandler(LlamaHandler):
    """Handler for Mistral models. Same structure as LLaMA."""
    pass
