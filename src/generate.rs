use std::sync::Arc;

use tch::{Device, IndexOp, Kind::Float, Tensor, nn::VarStore};

use crate::{constants::EOS_TOKEN, model::MiniGPT, tokenizer::Tokenizer};

pub struct MiniGPTGenerator {
    device: Device,
    model: Arc<MiniGPT>,
    tokenizer: Tokenizer,
    temperature: f64,
    max_length: usize,
}

impl MiniGPTGenerator {
    pub fn new(device: Device, checkpoint_path: &std::path::Path) -> MiniGPTGenerator {
        let mut var_store = VarStore::new(device);
        let model = Arc::new(MiniGPT::new(&var_store.root()));
        var_store
            .load(checkpoint_path)
            .expect("unable to load checkpoint");
        MiniGPTGenerator {
            device,
            model,
            tokenizer: Tokenizer::new(),
            temperature: 1.0,
            max_length: 2048,
        }
    }

    // pub fn set_temperature(&mut self, temp: f64) {
    //     self.temperature = temp;
    // }

    // pub fn set_max_length(&mut self, max_length: usize) {
    //     self.max_length = max_length;
    // }

    pub fn prompt(&self, prompt: &str) -> MiniGPTResponse {
        // Turn the prompt into a sequence of tokens
        let prompt_tokens = self
            .tokenizer
            .encode(&prompt)
            .expect("could not encode prompt (non-ASCII?)");
        // Turn the tokens into a 1D tensor [seq_len], then unsqueeze to [batch, seq_len]
        let prompt_tensor = Tensor::from_slice(&prompt_tokens)
            .to_kind(tch::Kind::Int64)
            .to_device(self.device)
            .unsqueeze(0);
        MiniGPTResponse {
            // Disable gradient tracking since we're not training now.
            _no_grad_guard: tch::no_grad_guard(),
            minigpt: self.model.clone(),
            response_length: 0,
            window_tensor: prompt_tensor,
            temperature: self.temperature,
            max_length: self.max_length,
        }
    }
}

pub struct MiniGPTResponse {
    _no_grad_guard: tch::NoGradGuard,
    minigpt: Arc<MiniGPT>,
    response_length: usize,
    window_tensor: Tensor,
    temperature: f64,
    max_length: usize,
}

impl Iterator for MiniGPTResponse {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        self.response_length += 1;
        // Get the logits (raw prediction) from the model as 3D [1, seq_len, VOCAB_SIZE]
        let model_logits = self.minigpt.forward(&self.window_tensor);
        // Get the final logit (predicts next token) [VOCAB_SIZE]
        let final_window_logit = model_logits.i((0, -1, ..));
        // Perform temp adjustment
        let temp_adjusted_logit = final_window_logit / self.temperature;
        // Softmax the vocab dimension
        let softmaxed_logit = temp_adjusted_logit.softmax(-1, Float);
        // Sample a token from the distribution
        let sampled_token_tensor = softmaxed_logit.multinomial(1, false);
        // Get the token as u8
        let sampled_token = u8::try_from(&sampled_token_tensor).expect("predicted token not a u8");
        // Break if it's EOS.
        if sampled_token == EOS_TOKEN || self.response_length >= self.max_length {
            return None;
        }
        // Trim the window to context max
        if self.window_tensor.size()[1] > 510 {
            let len = self.window_tensor.size()[1];
            self.window_tensor = self.window_tensor.narrow(1, len - 510, 510);
        }
        // Concatenate the sampled token to the window for the next loop.
        self.window_tensor = Tensor::cat(
            &[&self.window_tensor, &sampled_token_tensor.unsqueeze(0)],
            1,
        );
        Some(sampled_token as char)
    }
}
