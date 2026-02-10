# Future Plans

## Typed Tensor Newtypes
Wrap `tch::Tensor` in newtypes that encode dimensionality, e.g. `Tensor2(Tensor)`, `Tensor3(Tensor)`. This would make function signatures self-documenting (you'd see `Tensor3` instead of guessing the shape) and could catch shape mismatches at compile time. Significant wrapping effort — every tensor operation would need forwarding — but could make the codebase much more readable for Rust programmers unfamiliar with PyTorch conventions.

## KV Cache for Inference
Add key/value caching to the attention layers so that during autoregressive generation, only the new token's Q/K/V are computed each step. Requires threading cache state (a `Vec<(Tensor, Tensor)>` per layer) through `SingleHeadAttention`, `MultiHeadAttention`, `Transformer`, and `MiniGPT` forward methods. Prefill phase processes the full prompt and populates the cache; decode phase appends to it. Should give a significant speedup for generation — verify by comparing output and timing against the non-cached path.

## Top-k / Top-p Sampling
After temperature scaling and softmax, filter the probability distribution before sampling. Top-k keeps only the k highest-probability tokens; top-p (nucleus sampling) keeps the smallest set of tokens whose cumulative probability exceeds p. Both reduce the chance of sampling low-probability garbage tokens while preserving diversity. Implement as optional parameters on the generator.

## Validation Loss Tracking
Use the validation split from `DataSet` to compute validation loss periodically during training. Helps detect overfitting — if training loss keeps dropping but validation loss plateaus or rises, the model is memorizing rather than generalizing.

## Vocabulary-aware Tokenizer
Scan training data to build a vocabulary of only the characters that actually appear, rather than mapping all 128 ASCII values. Reduces embedding table size from 130 to ~70-80 rows. Minimal impact at current model size but would matter for a more optimized build.
