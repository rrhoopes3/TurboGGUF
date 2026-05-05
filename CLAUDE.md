# CLAUDE.md

Project memory for TurboGGUF. Add notes here that should persist across
Claude Code sessions (architectural reminders, related-work pointers, future
directions). User-facing docs go in README.md instead.

## Related work & ecosystem references

- **club-3090** — https://github.com/noonghunna/club-3090/tree/master
  Community / repo for RTX 3090 owners running local LLMs. Likely audience
  for TurboGGUF Q3_K_M / Q2_K builds (fits in 24 GB VRAM with KV-cache
  headroom). Worth engaging when publishing benchmark tables or rotated
  GGUFs to Hugging Face.

- **Gemma 4 Multi-Token Prediction (MTP)** — speculative decoding via a
  small drafter that shares input embeddings + reads the target's last-layer
  hidden states, then proposes 4–15 tokens which the target verifies in one
  pass. Up to ~3x throughput, exactly identical outputs (rejection sampling
  preserves the target distribution). Orthogonal to TurboGGUF's offline
  rotation: rotation lowers per-pass memory bandwidth via Q2/Q3 weights,
  MTP lowers the *number* of expensive passes. Stack both for the largest
  practical wins on consumer GPUs.
  - Overview: https://ai.google.dev/gemma/docs/mtp/overview
  - HF guide: https://ai.google.dev/gemma/docs/mtp/mtp
  - Blog: https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/

  Future direction: a "TurboGGUF + MTP" recipe doc showing rotated Q3_K_M
  target + matching assistant model (when available in GGUF) under
  llama.cpp's `--draft-model` / vLLM's speculative-decoding flag.
