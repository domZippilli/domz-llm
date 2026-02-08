/// A target context length.
pub const CONTEXT_LENGTH: u64 = 512;
/// The "vocabulary" size, which in this case is just the domain of ASCII characters, plus padding and end-of-sequence tokens.
pub const VOCAB_SIZE: i64 = 128 + 2; // 128 ASCII + PAD + EOS
/// The number of dimensions in the embedding space, which is set to the context length here.
pub const EMBEDDING_DIM: i64 = CONTEXT_LENGTH as i64;
/// The maximum sequence length is set to one less than the embedding dimension to allow for the EOS token.
pub const MAX_SEQUENCE_LENGTH: i64 = EMBEDDING_DIM - 1;
/// The number of "heads" (attention passes)
pub const HEADS: i64 = 8;
/// The "hidden" dimension for feed-forward network
pub const FFN_HIDDEN_DIM: i64 = EMBEDDING_DIM * 4;
/// Number of layers
pub const LAYERS: usize = 8;

/// Special token representing padding.
pub const PAD_TOKEN: u8 = 128;
/// Special token representing end-of-sequence.
pub const EOS_TOKEN: u8 = 129;

/// The fraction of the dataset to be used for training; the rest is used for validation.
pub const TRAIN_VALIDATION_SPLIT: f32 = 0.9;