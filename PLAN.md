# Plan: Build a Mini LLM from Scratch

## Context

Educational project to build a small GPT-style language model from scratch in Rust using `tch-rs` (bindings to PyTorch's C++ libtorch backend). The goal is to understand every component of an LLM by implementing it: attention, multi-head attention, transformer blocks, training loop, and inference with KV cache. The model will be trained on Python source code using a character-level tokenizer, and should run on an RTX 3080 Ti (12GB VRAM).

## Project Layout

```
domz-llm/
├── Cargo.toml          # tch dependency + binary targets
├── src/
│   ├── main.rs         # CLI entry point (train / generate subcommands)
│   ├── tokenizer.rs    # Character-level tokenizer (encode/decode)
│   ├── model.rs        # Transformer model (all the pieces)
│   ├── dataset.rs      # Data loading, batching, train/val split
│   ├── train.rs        # Training loop with checkpointing
│   └── generate.rs     # Inference / text generation with KV cache
├── checkpoints/        # Saved model weights
└── data/               # Training data (Python source files)
```

**Not a multi-service architecture** — it's a single Rust crate with modules and one binary that dispatches to train or generate via subcommands. Simple, graspable, everything visible.

## Model Specs (sized for 3080 Ti)

| Parameter | Value | Why |
|---|---|---|
| Vocab size | ~100 | Printable ASCII + special tokens (character-level) |
| Embedding dim | 512 | Good balance of capacity vs memory |
| Layers | 8 | Enough depth to learn patterns, small enough to train fast |
| Attention heads | 8 | 64 dims per head, standard ratio |
| FFN hidden dim | 2048 | 4x embedding dim (standard) |
| Context length | 512 | 512 characters — enough to see function-level patterns |
| Total params | ~25M | Trains in minutes-to-hours on a 3080 Ti |

## Dependencies

- **tch** (Rust bindings to libtorch) — provides tensors, autograd, nn modules, CUDA support
- **libtorch** (PyTorch C++ backend) — tch-rs downloads this automatically, or you can point it at an existing install via `LIBTORCH` env var
- PyTorch 2.9.0+cu128 is already installed; set `LIBTORCH_USE_PYTORCH=1` to reuse it

## Training Data

Concatenated Python source files. Options (in order of simplicity):
1. Scrape `.py` files from CPython stdlib or a well-known repo
2. Download a small subset from HuggingFace `datasets` (e.g., `bigcode/the-stack` filtered to Python)

We just need ~5-50MB of Python text. More data = better results, but even a few MB will show the model learning indentation, keywords, and basic structure.

## Development Steps

### Step 1: Project setup
- `cargo init` and add `tch` to `Cargo.toml`
- Verify libtorch links correctly and CUDA is accessible (`tch::Cuda::is_available()`)
- Download/prepare training data into `data/`

### Step 2: Character tokenizer (`tokenizer.rs`)
- Scan training data to build character vocabulary
- `encode(&str) -> Vec<i64>` and `decode(&[i64]) -> String`
- Special tokens: `<pad>`, `<eos>`
- This is intentionally trivial — the focus is the model

### Step 3: Dataset and batching (`dataset.rs`)
- Load encoded text as one long `Tensor`
- Chunk into sequences of `context_length`
- Train/val split (90/10)
- Return batches as `(input, target)` tensor pairs — no need for PyTorch's DataLoader, a simple batch iterator suffices

### Step 4: Build the model bottom-up (`model.rs`)
This is the core learning — each component maps to what we discussed:

1. **Token + Positional Embedding** — embed characters + learned position embeddings
2. **Single-head attention** — Q, K, V projections → scaled dot-product → causal mask → weighted sum
3. **Multi-head attention** — split into 8 heads, run attention per head, concatenate, project
4. **Feed-forward network (MLP)** — Linear → GELU → Linear
5. **Transformer block** — LayerNorm → MultiHeadAttn → residual → LayerNorm → MLP → residual
6. **Full model (MiniGPT)** — Embedding → 8x TransformerBlock → LayerNorm → linear head to vocab logits

Each component is a struct with a `forward(&self, xs: &Tensor) -> Tensor` method. Parameters are managed by tch's `nn::Path` / `nn::VarStore`.

We build and test each piece incrementally so you understand it before moving on.

### Step 5: Training loop (`train.rs`)
- Cross-entropy loss (next character prediction)
- AdamW optimizer via `nn::AdamW`, cosine learning rate schedule with warmup
- Gradient clipping via `VarStore::clip_grad_norm`
- Log training/validation loss
- Save checkpoints via `VarStore::save`
- Print sample generations during training so you can watch it improve

### Step 6: Train the model
- Start small (fewer layers/dims) to verify everything works
- Scale up to the full config
- Watch it go from random characters → valid Python keywords → indented blocks → function-like structures

### Step 7: Inference with KV cache (`generate.rs`)
- Load a checkpoint via `VarStore::load`
- Implement autoregressive generation: feed a prompt, sample next character, append, repeat
- **Implement KV cache** — this ties directly back to the EXO article discussion:
  - First pass (prefill): process full prompt, cache K and V at each layer
  - Subsequent passes (decode): only compute the new token, append to cache
  - You'll see the speedup firsthand
- Temperature and top-k/top-p sampling

## Build Order & What You'll Learn at Each Step

| Step | You'll understand... |
|---|---|
| Tokenizer | How text becomes numbers |
| Dataset | How sequences are batched for parallel training |
| Single-head attention | Q·K scoring, causal masking, weighted V aggregation |
| Multi-head attention | Why multiple heads, how dimensions split |
| Transformer block | Residual connections, layer norm, the full block pattern |
| Full model | How blocks stack, how logits become token probabilities |
| Training | Cross-entropy loss, backprop through the whole thing, gradient descent |
| KV cache inference | Why prefill and decode are different, what gets cached and why |

## Verification

- **Training**: loss should decrease steadily; after enough training, generated text should look like Python (proper indentation, keywords like `def`, `if`, `return`, colons, etc.)
- **KV cache**: generation with and without cache should produce identical output; cached version should be measurably faster
- **Sanity checks built in**: use `debug_assert!` for shape assertions throughout `model.rs` so dimensional mismatches are caught immediately during development
