use std::{f64, i64};

use crate::constants::{
    CONTEXT_LENGTH, EMBEDDING_DIM, FFN_HIDDEN_DIM, HEADS, MAX_SEQUENCE_LENGTH, LAYERS, VOCAB_SIZE
};
use tch::{
    Kind::{Bool, Float},
    Tensor,
    nn::{Embedding, LayerNorm, Linear, Path, embedding, layer_norm, linear},
};

struct Embeddings {
    token: Embedding,
    position: Embedding,
}

impl Embeddings {
    pub fn new(vs: &Path) -> Self {
        let token = embedding(vs / "token", VOCAB_SIZE, EMBEDDING_DIM, Default::default());
        let position = embedding(
            vs / "position",
            MAX_SEQUENCE_LENGTH,
            EMBEDDING_DIM,
            Default::default(),
        );
        Embeddings { token, position }
    }

    /// Takes a tensor of token IDs with shape `[batch_size, seq_len]` and returns                    
    /// a tensor of shape `[batch_size, seq_len, EMBEDDING_DIM]` where each token ID
    /// has been replaced by the sum of its token embedding and its position embedding.  
    pub fn forward(&self, input: &tch::Tensor) -> tch::Tensor {
        // TODO(domz): Understand this lookup a bit better. Why are the dimensions these specifically?
        // Look up token embeddings: [batch_size, seq_len] -> [batch_size, seq_len, 512]
        let token_embeds = input.apply(&self.token);
        // Get the sequence length from the input's second dimension
        let seq_length = input.size()[1];
        // Create position indices [0, 1, 2, ..., seq_len-1] on the same device as input
        let position_ids = tch::Tensor::arange(seq_length, (tch::Kind::Int64, input.device()))
            // Add batch dimension: [seq_len] -> [1, seq_len]
            .unsqueeze(0)
            // Replicate for each item in the batch: [1, seq_len] -> [batch_size, seq_len]
            .expand(&input.size(), true);
        // Look up position embeddings: [batch_size, seq_len] -> [batch_size, seq_len, 512]
        let position_embeds = position_ids.apply(&self.position);
        // Combine: each token is now represented by token_embedding + position_embedding
        token_embeds + position_embeds
    }
}

/// Single-head attention mechanism.                                                            
/// Takes an input of shape [batch_size, seq_len, embed_dim] and produces
/// an output of the same shape, where each position's output is a weighted                     
/// combination of other positions' values, with weights determined by                          
/// query-key compatibility scores.
struct SingleHeadAttention {
    query: Linear, // Projects embed_dim -> head_dim
    key: Linear,   // Projects embed_dim -> head_dim
    value: Linear, // Projects embed_dim -> head_dim
    head_dim: i64,
}

impl SingleHeadAttention {
    /// Creates Q, K, V projection layers. head_dim = embed_dim / num_heads (512/8 = 64).
    ///   - Q (Query): "what am I looking for?"
    ///   - K (Key): "what do I contain?"
    ///   - V (Value): "what information do I provide?"

    pub fn new(vs: &Path) -> Self {
        let head_dim = EMBEDDING_DIM / HEADS;
        // TODO(domz): "applies an affine linear transform" -- understand this better?
        let query = linear(vs / "query", EMBEDDING_DIM, head_dim, Default::default());
        let key = linear(vs / "key", EMBEDDING_DIM, head_dim, Default::default());
        let value = linear(vs / "value", EMBEDDING_DIM, head_dim, Default::default());

        SingleHeadAttention {
            query,
            key,
            value,
            head_dim,
        }
    }

    /// Claude's steps:
    /// 1. Project input into Q, K, V          (each: [batch, seq_len, head_dim])
    /// 2. Compute attention scores: Q @ K^T    (shape: [batch, seq_len, seq_len])
    /// 3. Scale by sqrt(head_dim)
    /// 4. Apply causal mask (future positions -> -infinity)
    /// 5. Softmax over the last dimension      (scores sum to 1 per row)
    /// 6. Weighted sum: scores @ V             (shape: [batch, seq_len, head_dim])
    ///
    /// Dom's docstring:
    /// Takes an input of shape [batch_size, seq_len, embed_dim] and produces
    /// a head-sized output of [batch_size, seq_len, head_dim].
    pub fn forward(&self, input: &tch::Tensor) -> Tensor {
        // Apply the linear transform to reduce dimensions to head size
        // [batch, seq_len, 64]
        let q_embeds = input.apply(&self.query);
        // [batch, seq_len, 64]
        let k_embeds = input.apply(&self.key);
        // [batch, seq_len, 64]
        let v_embeds = input.apply(&self.value);
        // Attention scores are Q @ K^T -- dot product of Q and transposed K
        // Transpose to line up [batch, seq_len, seq_len]
        let k_embeds_transposed = k_embeds.transpose(-2, -1);
        // Matmul for [M, P] and then...
        // divide by sqrt(head_dim) to scale (for less sharp distribution heading into softmax, creating larger gradients which improves learning speed)
        let attention_scores =
            q_embeds.matmul(&k_embeds_transposed) / f64::sqrt(self.head_dim as f64);
        // Create a triangular matrix mask with the upper triangle masked to -infinity
        let future_masked_attention_scores = attention_scores.masked_fill(
            // TODO(domz): What is ones_like really doing???
            &attention_scores.ones_like().triu(1).to_kind(Bool),
            f64::NEG_INFINITY,
        );
        // Softmax the attention scores
        let softmaxed_attention_scores = future_masked_attention_scores.softmax(-1, Float);
        // Produce a tensor with weighted sums (shape: [batch, seq_len, head_dim])
        softmaxed_attention_scores.matmul(&v_embeds)
    }
}

struct MultiHeadAttention {
    heads: Vec<SingleHeadAttention>,
    output_projection: Linear,
}

impl MultiHeadAttention {
    pub fn new(vs: &Path) -> MultiHeadAttention {
        let mut heads = Vec::with_capacity(HEADS as usize);
        for i in 0..(HEADS as usize) {
            heads.push(SingleHeadAttention::new(&(vs / format!("head_{}", i))));
        }
        let output_projection = linear(
            vs / "output",
            EMBEDDING_DIM,
            EMBEDDING_DIM,
            Default::default(),
        );
        MultiHeadAttention {
            heads,
            output_projection,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        // Get all the outputs from the heads: Vec of [batch, seq_len, head_dim]
        let outputs: Vec<Tensor> = self.heads.iter().map(|head| head.forward(input)).collect();
        // Concatenate along the last dimension to combine head outputs: [batch, seq_len, head_dim * num_heads]
        let concatenated = Tensor::cat(&outputs, -1);
        concatenated.apply(&self.output_projection)
    }
}

struct FeedForward {
    project_up: Linear,
    project_down: Linear,
}

impl FeedForward {
    pub fn new(vs: &Path) -> FeedForward {
        FeedForward {
            project_up: linear(vs / "up", EMBEDDING_DIM, FFN_HIDDEN_DIM, Default::default()),
            project_down: linear(
                vs / "down",
                FFN_HIDDEN_DIM,
                EMBEDDING_DIM,
                Default::default(),
            ),
        }
    }

    //   Attention lets each token gather information from other tokens. But
    //   after gathering, each token needs to process that information
    //   individually. That's what the "Multi-layer Perceptron"(!) or MLP does —
    //   it's applied to each position independently (same weights, no
    //   cross-position interaction).
    pub fn forward(&self, input: &Tensor) -> Tensor {
        // increase the dimensions for more "scratch space"
        let expanded_embeddings = input.apply(&self.project_up);
        // run Gaussian error linear units function
        // TODO(domz): Learn the "why" of GELU
        let gelu_activated = expanded_embeddings.gelu("none");
        // Shrink the tensor back down
        gelu_activated.apply(&self.project_down)
    }
}

struct Transformer {
    input_layernorm: LayerNorm,
    attended_layernorm: LayerNorm,
    attention_layer: MultiHeadAttention,
    ffn: FeedForward,
}

impl Transformer {
    pub fn new(vs: &Path) -> Transformer {
        let input_layernorm = layer_norm(
            vs / "input_layernorm",
            vec![EMBEDDING_DIM],
            Default::default(),
        );
        let attended_layernorm = layer_norm(
            vs / "attended_layernorm",
            vec![EMBEDDING_DIM],
            Default::default(),
        );
        let attention_layer = MultiHeadAttention::new(&(vs / "attention"));
        let ffn = FeedForward::new(&(vs / "ffn"));
        Transformer {
            input_layernorm,
            attended_layernorm,
            attention_layer,
            ffn,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        // normalize input and build the attention layer
        let normalized_input = input.apply(&self.input_layernorm);
        let attention = self.attention_layer.forward(&normalized_input);
        let attended = input + attention; // "residual connection" 1

        // normalize attended and backprop through FFN
        let normalized_attended = attended.apply(&self.attended_layernorm);
        let perceptron = self.ffn.forward(&normalized_attended);
        attended + perceptron // "residual connection" 2
    }
}

pub struct MiniGPT {
    embeddings: Embeddings,
    layers: Vec<Transformer>,
    final_layernorm: LayerNorm,
    output_head: Linear,
}

impl MiniGPT {
    pub fn new (vs: &Path) -> MiniGPT {
        let embeddings = Embeddings::new(&(vs / "embeddings"));
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..(LAYERS) {
            layers.push(Transformer::new(&(vs / format!("transformer_{}", i))));
        };
        let final_layernorm = layer_norm(
            vs / "final_layernorm",
            vec![EMBEDDING_DIM],
            Default::default(),
        );
        let output_head = linear(vs / "output", EMBEDDING_DIM, VOCAB_SIZE, Default::default());
        MiniGPT { embeddings, layers, final_layernorm, output_head }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let mut embeddings = self.embeddings.forward(input);
        for transformer in &self.layers {
            embeddings = transformer.forward(&embeddings);
        };
        embeddings.apply(&self.final_layernorm).apply(&self.output_head)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor, nn};

    #[test]
    fn test_forward_output_shape() {
        let vs = nn::VarStore::new(Device::Cpu);
        let model = Embeddings::new(&vs.root());
        // Batch of 2 sequences, each 4 tokens long
        let input = Tensor::from_slice2(&[&[72i64, 101, 108, 108], &[100, 101, 102, 32]]);
        let output = model.forward(&input);
        assert_eq!(output.size(), &[2, 4, EMBEDDING_DIM as i64]);
    }

    #[test]
    fn test_forward_single_token() {
        let vs = nn::VarStore::new(Device::Cpu);
        let model = Embeddings::new(&vs.root());
        // Batch of 1, sequence of 1 token
        let input = Tensor::from_slice2(&[&[65i64]]);
        let output = model.forward(&input);
        assert_eq!(output.size(), &[1, 1, EMBEDDING_DIM as i64]);
    }

    #[test]
    fn test_forward_full_sequence() {
        let vs = nn::VarStore::new(Device::Cpu);
        let model = Embeddings::new(&vs.root());
        // Batch of 1, full sequence length, all valid token IDs (mod vocab size)
        let input = Tensor::arange(MAX_SEQUENCE_LENGTH as i64, (Kind::Int64, Device::Cpu))
            .remainder(VOCAB_SIZE as i64)
            .unsqueeze(0);
        let output = model.forward(&input);
        assert_eq!(
            output.size(),
            &[1, MAX_SEQUENCE_LENGTH as i64, EMBEDDING_DIM as i64]
        );
    }

    #[test]
    fn test_same_token_different_positions() {
        let vs = nn::VarStore::new(Device::Cpu);
        let model = Embeddings::new(&vs.root());
        // Same token (65) at every position — outputs should differ due to position embeddings
        let input = Tensor::ones(&[1, 4], (Kind::Int64, Device::Cpu)) * 65;
        let output = model.forward(&input);
        let pos0 = output.get(0).get(0);
        let pos1 = output.get(0).get(1);
        // Position embeddings make these different even though the token is the same
        let diff = (&pos0 - &pos1).abs().sum(Kind::Float);
        assert!(f64::try_from(&diff).unwrap() > 0.0);
    }

    // -- SingleHeadAttention tests --

    /// Helper: create a random float input tensor shaped [batch, seq_len, EMBEDDING_DIM]
    fn random_embed_input(batch: i64, seq_len: i64) -> Tensor {
        Tensor::randn(&[batch, seq_len, EMBEDDING_DIM], (Float, Device::Cpu))
    }

    #[test]
    fn test_single_head_output_shape() {
        let vs = nn::VarStore::new(Device::Cpu);
        let head = SingleHeadAttention::new(&vs.root());
        let input = random_embed_input(2, 8);
        let output = head.forward(&input);
        let head_dim = EMBEDDING_DIM / HEADS;
        assert_eq!(output.size(), &[2, 8, head_dim]);
    }

    #[test]
    fn test_single_head_single_token() {
        let vs = nn::VarStore::new(Device::Cpu);
        let head = SingleHeadAttention::new(&vs.root());
        let input = random_embed_input(1, 1);
        let output = head.forward(&input);
        let head_dim = EMBEDDING_DIM / HEADS;
        assert_eq!(output.size(), &[1, 1, head_dim]);
    }

    #[test]
    fn test_single_head_causal_masking() {
        // Verify that changing a future token doesn't affect an earlier position's output
        let vs = nn::VarStore::new(Device::Cpu);
        let head = SingleHeadAttention::new(&vs.root());

        let input_a = Tensor::randn(&[1, 4, EMBEDDING_DIM], (Float, Device::Cpu));
        let output_a = head.forward(&input_a);

        // Modify position 3 (the last token) — positions 0, 1, 2 should be unaffected
        let mut input_b = input_a.copy();
        let noise = Tensor::randn(&[1, 1, EMBEDDING_DIM], (Float, Device::Cpu));
        input_b.narrow(1, 3, 1).copy_(&noise);
        let output_b = head.forward(&input_b);

        // Positions 0, 1, 2 should be identical
        for pos in 0..3 {
            let a = output_a.get(0).get(pos);
            let b = output_b.get(0).get(pos);
            let diff = (&a - &b).abs().sum(Float);
            assert!(
                f64::try_from(&diff).unwrap() < 1e-5,
                "Position {} changed when only a future token was modified",
                pos
            );
        }

        // Position 3 should be different (it can see itself, which changed)
        let a3 = output_a.get(0).get(3);
        let b3 = output_b.get(0).get(3);
        let diff3 = (&a3 - &b3).abs().sum(Float);
        assert!(
            f64::try_from(&diff3).unwrap() > 1e-5,
            "Position 3 should have changed"
        );
    }

    // -- MultiHeadAttention tests --

    #[test]
    fn test_multi_head_output_shape() {
        let vs = nn::VarStore::new(Device::Cpu);
        let mha = MultiHeadAttention::new(&vs.root());
        let input = random_embed_input(2, 8);
        let output = mha.forward(&input);
        // Output should be back to full embed_dim after projection
        assert_eq!(output.size(), &[2, 8, EMBEDDING_DIM]);
    }

    #[test]
    fn test_multi_head_single_token() {
        let vs = nn::VarStore::new(Device::Cpu);
        let mha = MultiHeadAttention::new(&vs.root());
        let input = random_embed_input(1, 1);
        let output = mha.forward(&input);
        assert_eq!(output.size(), &[1, 1, EMBEDDING_DIM]);
    }

    #[test]
    fn test_multi_head_causal_masking() {
        let vs = nn::VarStore::new(Device::Cpu);
        let mha = MultiHeadAttention::new(&vs.root());

        let input_a = Tensor::randn(&[1, 4, EMBEDDING_DIM], (Float, Device::Cpu));
        let output_a = mha.forward(&input_a);

        // Modify position 3 — earlier positions should be unaffected
        let input_b = input_a.copy();
        let noise = Tensor::randn(&[1, 1, EMBEDDING_DIM], (Float, Device::Cpu));
        input_b.narrow(1, 3, 1).copy_(&noise);
        let output_b = mha.forward(&input_b);

        for pos in 0..3 {
            let a = output_a.get(0).get(pos);
            let b = output_b.get(0).get(pos);
            let diff = (&a - &b).abs().sum(Float);
            assert!(
                f64::try_from(&diff).unwrap() < 1e-5,
                "Position {} changed when only a future token was modified",
                pos
            );
        }
    }

    // -- FeedForward tests --

    #[test]
    fn test_ffn_output_shape() {
        let vs = nn::VarStore::new(Device::Cpu);
        let ffn = FeedForward::new(&vs.root());
        let input = random_embed_input(2, 8);
        let output = ffn.forward(&input);
        // FFN should preserve shape: [batch, seq_len, EMBEDDING_DIM]
        assert_eq!(output.size(), &[2, 8, EMBEDDING_DIM]);
    }

    #[test]
    fn test_ffn_single_token() {
        let vs = nn::VarStore::new(Device::Cpu);
        let ffn = FeedForward::new(&vs.root());
        let input = random_embed_input(1, 1);
        let output = ffn.forward(&input);
        assert_eq!(output.size(), &[1, 1, EMBEDDING_DIM]);
    }

    #[test]
    fn test_ffn_position_independence() {
        // FFN applies independently per position — changing one position
        // should not affect any other position's output
        let vs = nn::VarStore::new(Device::Cpu);
        let ffn = FeedForward::new(&vs.root());

        let input_a = random_embed_input(1, 4);
        let output_a = ffn.forward(&input_a);

        let input_b = input_a.copy();
        let noise = Tensor::randn(&[1, 1, EMBEDDING_DIM], (Float, Device::Cpu));
        input_b.narrow(1, 2, 1).copy_(&noise);
        let output_b = ffn.forward(&input_b);

        // Positions 0, 1, 3 should be identical (only position 2 changed)
        for pos in [0, 1, 3] {
            let a = output_a.get(0).get(pos);
            let b = output_b.get(0).get(pos);
            let diff = (&a - &b).abs().sum(Float);
            assert!(
                f64::try_from(&diff).unwrap() < 1e-5,
                "Position {} changed when only position 2 was modified",
                pos
            );
        }

        // Position 2 should be different
        let a2 = output_a.get(0).get(2);
        let b2 = output_b.get(0).get(2);
        let diff2 = (&a2 - &b2).abs().sum(Float);
        assert!(
            f64::try_from(&diff2).unwrap() > 1e-5,
            "Position 2 should have changed"
        );
    }

    // -- Transformer block tests --

    #[test]
    fn test_transformer_output_shape() {
        let vs = nn::VarStore::new(Device::Cpu);
        let block = Transformer::new(&vs.root());
        let input = random_embed_input(2, 8);
        let output = block.forward(&input);
        // Transformer block preserves shape
        assert_eq!(output.size(), &[2, 8, EMBEDDING_DIM]);
    }

    #[test]
    fn test_transformer_single_token() {
        let vs = nn::VarStore::new(Device::Cpu);
        let block = Transformer::new(&vs.root());
        let input = random_embed_input(1, 1);
        let output = block.forward(&input);
        assert_eq!(output.size(), &[1, 1, EMBEDDING_DIM]);
    }

    #[test]
    fn test_transformer_causal_masking() {
        let vs = nn::VarStore::new(Device::Cpu);
        let block = Transformer::new(&vs.root());

        let input_a = Tensor::randn(&[1, 4, EMBEDDING_DIM], (Float, Device::Cpu));
        let output_a = block.forward(&input_a);

        // Modify position 3 — earlier positions should be unaffected
        let input_b = input_a.copy();
        let noise = Tensor::randn(&[1, 1, EMBEDDING_DIM], (Float, Device::Cpu));
        input_b.narrow(1, 3, 1).copy_(&noise);
        let output_b = block.forward(&input_b);

        for pos in 0..3 {
            let a = output_a.get(0).get(pos);
            let b = output_b.get(0).get(pos);
            let diff = (&a - &b).abs().sum(Float);
            assert!(
                f64::try_from(&diff).unwrap() < 1e-5,
                "Position {} changed when only a future token was modified",
                pos
            );
        }
    }

    #[test]
    fn test_transformer_residual_passthrough() {
        // With freshly initialized (near-zero) weights, the residual connections
        // should mean the output is close to the input
        let vs = nn::VarStore::new(Device::Cpu);
        let block = Transformer::new(&vs.root());
        let input = random_embed_input(1, 4);
        let output = block.forward(&input);
        let diff = (&input - &output).abs().mean(Float);
        // Not exactly equal (weights aren't exactly zero), but should be in the same ballpark
        assert!(
            f64::try_from(&diff).unwrap() < 5.0,
            "With fresh weights, residual connections should keep output near input"
        );
    }

    // -- MiniGPT full model tests --

    #[test]
    fn test_minigpt_output_shape() {
        let vs = nn::VarStore::new(Device::Cpu);
        let model = MiniGPT::new(&vs.root());
        // Batch of 2, sequence of 8 token IDs
        let input = Tensor::from_slice2(&[
            &[72i64, 101, 108, 108, 111, 32, 87, 111],
            &[100, 101, 102, 32, 102, 111, 111, 40],
        ]);
        let output = model.forward(&input);
        // Output should be [batch, seq_len, VOCAB_SIZE] — logits over vocabulary
        assert_eq!(output.size(), &[2, 8, VOCAB_SIZE]);
    }

    #[test]
    fn test_minigpt_single_token() {
        let vs = nn::VarStore::new(Device::Cpu);
        let model = MiniGPT::new(&vs.root());
        let input = Tensor::from_slice2(&[&[65i64]]);
        let output = model.forward(&input);
        assert_eq!(output.size(), &[1, 1, VOCAB_SIZE]);
    }

    #[test]
    fn test_minigpt_causal_masking() {
        let vs = nn::VarStore::new(Device::Cpu);
        let model = MiniGPT::new(&vs.root());

        let input_a = Tensor::from_slice2(&[&[72i64, 101, 108, 108]]);
        let output_a = model.forward(&input_a);

        // Change the last token only
        let input_b = Tensor::from_slice2(&[&[72i64, 101, 108, 40]]);
        let output_b = model.forward(&input_b);

        // Positions 0, 1, 2 should be identical
        for pos in 0..3 {
            let a = output_a.get(0).get(pos);
            let b = output_b.get(0).get(pos);
            let diff = (&a - &b).abs().sum(Float);
            assert!(
                f64::try_from(&diff).unwrap() < 1e-4,
                "Position {} changed when only a future token was modified",
                pos
            );
        }

        // Position 3 should differ
        let a3 = output_a.get(0).get(3);
        let b3 = output_b.get(0).get(3);
        let diff3 = (&a3 - &b3).abs().sum(Float);
        assert!(
            f64::try_from(&diff3).unwrap() > 1e-5,
            "Position 3 should have changed"
        );
    }

    #[test]
    fn test_minigpt_logits_are_not_uniform() {
        // Even with random weights, the model should produce non-uniform
        // logits across the vocabulary (not all the same value)
        let vs = nn::VarStore::new(Device::Cpu);
        let model = MiniGPT::new(&vs.root());
        let input = Tensor::from_slice2(&[&[72i64, 101, 108]]);
        let output = model.forward(&input);
        // Check that logits at position 0 have some variance
        let logits = output.get(0).get(0); // [VOCAB_SIZE]
        let std_dev = f64::try_from(&logits.std(false)).unwrap();
        assert!(
            std_dev > 1e-6,
            "Logits should not be uniform across vocabulary"
        );
    }
}
