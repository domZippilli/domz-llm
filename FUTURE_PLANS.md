# Future Plans

## Typed Tensor Newtypes
Wrap `tch::Tensor` in newtypes that encode dimensionality, e.g. `Tensor2(Tensor)`, `Tensor3(Tensor)`. This would make function signatures self-documenting (you'd see `Tensor3` instead of guessing the shape) and could catch shape mismatches at compile time. Significant wrapping effort — every tensor operation would need forwarding — but could make the codebase much more readable for Rust programmers unfamiliar with PyTorch conventions.

## Vocabulary-aware Tokenizer
Scan training data to build a vocabulary of only the characters that actually appear, rather than mapping all 128 ASCII values. Reduces embedding table size from 130 to ~70-80 rows. Minimal impact at current model size but would matter for a more optimized build.
