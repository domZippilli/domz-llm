# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational project building a GPT-style language model (~25M params) from scratch in Rust using `tch-rs` (bindings to libtorch/PyTorch C++ backend). Character-level tokenizer trained on Python source code. Target hardware: RTX 3080 Ti (12GB VRAM).

**Status**: Planning phase — only `PLAN.md` exists, no code implemented yet. See `PLAN.md` for the full specification.

## Architecture

Single Rust crate with one binary. `src/main.rs` provides a CLI with `train` and `generate` subcommands. The remaining modules:

- `tokenizer.rs` — Character-level encode/decode over printable ASCII + `<pad>`, `<eos>`
- `dataset.rs` — Loads encoded text as one tensor, chunks into 512-token sequences, 90/10 train/val split, returns `(input, target)` batch pairs
- `model.rs` — Transformer built bottom-up: embeddings → single-head attention → multi-head attention (8 heads) → FFN (GELU) → transformer block (pre-norm residual) → full MiniGPT (8 layers). Each component is a struct with a `forward` method; parameters managed via `nn::VarStore`
- `train.rs` — Cross-entropy loss, AdamW + cosine LR with warmup, gradient clipping via `VarStore::clip_grad_norm`, periodic checkpointing (`VarStore::save`) and sample generation
- `generate.rs` — Autoregressive generation with KV cache (prefill/decode phases), temperature and top-k/top-p sampling, loads checkpoints via `VarStore::load`

Model specs: vocab ~100, embed_dim 512, 8 layers, 8 heads (64 dim/head), FFN hidden 2048, context 512.

## Commands

```bash
# Build
cargo build --release

# Training
cargo run --release -- train --data data/

# Inference
cargo run --release -- generate --checkpoint checkpoints/latest.safetensors --prompt "def "

# Run tests
cargo test

# Verify CUDA is accessible
cargo run --release -- 2>&1 | head   # should not error on libtorch init
```

## libtorch Setup

`tch-rs` needs libtorch. Since PyTorch 2.9.0+cu128 is already installed, set:
```bash
export LIBTORCH_USE_PYTORCH=1
```
Alternatively, `tch` can download libtorch automatically — see the [tch-rs README](https://github.com/LaurentMazare/tch-rs) for options.

## Development Order

Build and test components incrementally (see PLAN.md Steps 1-7):
1. Project setup — `cargo init`, verify libtorch + CUDA linking
2. Character tokenizer
3. Dataset and batching (simple batch iterator, no DataLoader needed)
4. Model components bottom-up (with `debug_assert!` shape checks at each layer)
5. Training loop
6. Train small config first, then scale to full
7. Inference with KV cache

## Verification

- Training loss should decrease steadily; generated text should progress from random → Python keywords → indented blocks → function-like structures
- KV cache and non-cached generation must produce identical output; cached version should be measurably faster
- `debug_assert!` shape checks throughout `model.rs` catch dimensional mismatches in dev builds
