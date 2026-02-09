# Comprehension Test: Building a GPT from Scratch

Answer these questions based on what you learned building this project. The goal isn't to recite code — it's to confirm you could explain these concepts to someone else.

---

### 1. Tokenization

This project uses a character-level tokenizer rather than something like BPE (byte-pair encoding) used by real LLMs. What trade-off does that create? In other words, what do you gain and what do you lose compared to subword tokenization?

---

### 2. Token vs. Positional Embeddings

The `Embeddings` layer produces its output by *adding* the token embedding and the position embedding together. Why addition, rather than concatenation? What would go wrong (or change) if you concatenated them instead?

---

### 3. The Shape Journey

Trace the tensor shape through a single forward pass of MiniGPT. Starting from a raw input of shape `[batch_size, seq_len]`, what is the shape after each of these stages?

- a) After `Embeddings.forward()`
- b) After a single `SingleHeadAttention.forward()`
- c) After `MultiHeadAttention.forward()`
- d) After `FeedForward.forward()`
- e) After `MiniGPT.forward()` (the final output)

---

### 4. Q, K, V — In Your Own Words

Explain what the Query, Key, and Value projections each represent, without using the words "query", "key", or "value". Use an analogy or a plain-English description of what role each plays in computing attention.

---

### 5. Attention Score Scaling

In `SingleHeadAttention.forward()`, the raw attention scores (`Q @ K^T`) are divided by `sqrt(head_dim)`. What problem does this solve? What would happen during training if you removed this scaling?

---

### 6. Causal Masking

The causal mask sets upper-triangle entries to negative infinity *before* softmax. Two-part question:

- a) Why negative infinity specifically (as opposed to, say, zero)?
- b) The test `test_single_head_causal_masking` verifies this by modifying position 3 and checking that positions 0-2 are unaffected. Why is this property essential for a language model that generates text left-to-right?

---

### 7. Why Multiple Heads?

`MultiHeadAttention` runs 8 separate `SingleHeadAttention` instances, each projecting to 64 dimensions instead of one head projecting to 512. The outputs are concatenated and then projected back to 512.

Why is this better than a single attention head with the full 512-dimensional Q, K, V? What does each head "specialize" in (conceptually)?

---

### 8. The Feed-Forward Network

The FFN expands from 512 to 2048 dimensions, applies GELU, then projects back to 512.

- a) The test `test_ffn_position_independence` proves that the FFN operates on each position independently. Why is this an important property, given that attention already handles cross-position interaction?
- b) Why expand to a *larger* dimension and then shrink back, rather than just applying a 512->512 linear layer?

---

### 9. Residual Connections

In `Transformer.forward()`, the pattern is:

```
attended = input + attention(layernorm(input))
output   = attended + ffn(layernorm(attended))
```

- a) What problem do residual connections solve in deep networks?
- b) The test `test_transformer_residual_passthrough` checks that with freshly initialized weights, output is close to input. Why does this property matter for training?

---

### 10. Pre-Norm vs. Post-Norm

This model applies LayerNorm *before* the attention/FFN sublayers (pre-norm), not after. This is a deliberate architectural choice. What advantage does pre-norm have over post-norm, especially during training?

---

### 11. The Output Head

`MiniGPT` ends with a linear layer that projects from `[batch, seq_len, EMBEDDING_DIM]` to `[batch, seq_len, VOCAB_SIZE]`. These raw outputs are called "logits."

- a) What do the logits represent, and how do you turn them into a probability distribution over the vocabulary?
- b) During training, the code computes `cross_entropy_for_logits` against the target. In plain terms, what does cross-entropy loss measure?

---

### 12. Input/Target Relationship

In `dataset.rs`, for a chunk of tokens `[A, B, C, D, E]`, the input is `[A, B, C, D]` and the target is `[B, C, D, E]` — shifted by one position. Why this offset? What is the model being asked to learn at each position?

---

### 13. Batch Size = 1

Look at `train_step`: the input tensor is unsqueezed to add a batch dimension of 1 (`input.unsqueeze(0)`), meaning each training step processes a single sequence. Real LLMs train with large batch sizes (hundreds or thousands of sequences at once). What are the two main benefits of larger batch sizes?

---

### 14. The Optimizer

The trainer uses Adam (a variant of SGD). In one or two sentences, what does Adam do differently from plain stochastic gradient descent? Why does this tend to help?

---

### 15. Gradient Clipping

The PLAN.md mentions gradient clipping via `VarStore::clip_grad_norm`. This isn't yet implemented in the current training loop. What problem does gradient clipping prevent, and when is it most likely to occur during training?

---

### 16. Autoregressive Generation

During inference, the model generates text one token at a time: encode the prompt, run it through the model, sample from the *last position's* logits, append the new token, and repeat.

Why only the last position's logits? What do the logits at earlier positions represent, and why aren't they useful during generation?

---

### 17. KV Cache — The Big Idea

The PLAN.md describes KV caching with "prefill" and "decode" phases. Without a cache, generating N tokens requires running the full model N times, each time over a growing sequence.

- a) What exactly gets cached, and why don't you need to cache Q?
- b) During the decode phase (generating token N+1), what is the shape of the Q tensor versus the shape of the cached K and V tensors?

---

### 18. Temperature and Sampling

The plan mentions temperature and top-k/top-p sampling for generation.

- a) Temperature is applied by dividing logits by a temperature value before softmax. What happens when temperature approaches 0? What about when it's very high (e.g., 10)?
- b) Top-k sampling restricts the choice to the k highest-probability tokens. Why would you want to exclude low-probability tokens rather than just letting the probabilities naturally handle it?

---

### 19. Connecting It All

If you doubled `EMBEDDING_DIM` from 512 to 1024 (keeping everything else the same), list three specific consequences this would have on the model. Think about parameter count, memory usage, and the shapes flowing through the network.

---

### 20. The Mechanical Sympathy Question

You built this to develop "mechanical sympathy" for LLMs. Name one concrete insight you gained from this project that changes how you think about using or prompting an LLM in practice. There's no wrong answer — this is about what clicked for you.
