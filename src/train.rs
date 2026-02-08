use std::f64;

use tch::{
    Device, Tensor,
    nn::{Adam, Optimizer, OptimizerConfig, VarStore},
};

use crate::{constants::VOCAB_SIZE, dataset::DataSet, model::MiniGPT};

struct MiniGPTTrainer {
    var_store: VarStore,
    model: MiniGPT,
    optimizer: Optimizer,
    dataset: DataSet,
}

impl MiniGPTTrainer {
    pub fn new(device: Device, dataset: DataSet) -> MiniGPTTrainer {
        let var_store = VarStore::new(device);
        let model = MiniGPT::new(&var_store.root());
        let optimizer = Adam::default()
            .build(&var_store, 1e-4)
            .expect("failed to create Adam optimizer");
        MiniGPTTrainer {
            var_store,
            model,
            optimizer,
            dataset,
        }
    }

    // Train the model for the given epochs.
    pub fn train(&mut self, epochs: usize) {
        for epoch in 0..epochs {
            let loss = self.train_epoch();
            println!("epoch: {}, loss: {}", epoch, loss);
        }
    }

    // Perform one epoch.
    fn train_epoch(&mut self) -> f64 {
        // Do some averaging of the loss over each input
        let mut total_loss = 0.0;
        let mut inputs = 0;
        let training_data = self.dataset.get_training_dataset();
        for batch in training_data.into_iter() {
            for (input, target) in batch.iter() {
                inputs += 1;
                total_loss += self.train_step(input, target);
            }
        }
        total_loss / inputs as f64
    }

    // Perform one training step.
    fn train_step(&mut self, input: &Tensor, target: &Tensor) -> f64 {
        // unsqueeze input to add dimension, forward through model, get raw predictions (logits1)
        let logits = self.model.forward(&input.unsqueeze(0)); // [1, seq_len, VOCAB_SIZE]
        // compute loss, logits needs to be flattened to [seq_len, VOCAB_SIZE]
        let loss = logits
            .view([-1, VOCAB_SIZE])
            .cross_entropy_for_logits(target);
        // backprop gradients, apply Adam update rule (adjust according to volatility/confidence)
        self.optimizer.backward_step(&loss);
        f64::try_from(&loss).expect("couldn't compute f64 from loss tensor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Tokenizer;
    use std::io::Write;

    fn setup_trainer() -> (MiniGPTTrainer, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..3 {
            let path = dir.path().join(format!("test_{}.py", i));
            let mut f = std::fs::File::create(path).unwrap();
            writeln!(f, "def foo_{}():\n    return {}", i, i).unwrap();
        }
        let dataset = DataSet::new(dir.path().to_string_lossy().to_string(), Tokenizer::new());
        let trainer = MiniGPTTrainer::new(Device::Cpu, dataset);
        (trainer, dir) // keep dir alive so temp files aren't deleted
    }

    #[test]
    fn test_train_step_loss_is_finite() {
        let (mut trainer, _dir) = setup_trainer();
        // Create a simple synthetic input/target pair
        let input = Tensor::from_slice(&[100i64, 101, 102, 32, 102, 111, 111, 40]);
        let target = Tensor::from_slice(&[101i64, 102, 32, 102, 111, 111, 40, 41]);
        let loss = trainer.train_step(&input, &target);
        assert!(loss.is_finite(), "Loss should be a finite number, got {}", loss);
        assert!(loss > 0.0, "Cross-entropy loss should be positive, got {}", loss);
    }

    #[test]
    fn test_train_step_loss_decreases() {
        let (mut trainer, _dir) = setup_trainer();
        // Train on the same input repeatedly — loss should decrease
        let input = Tensor::from_slice(&[100i64, 101, 102, 32, 102, 111, 111, 40]);
        let target = Tensor::from_slice(&[101i64, 102, 32, 102, 111, 111, 40, 41]);

        let first_loss = trainer.train_step(&input, &target);
        let mut last_loss = first_loss;
        for _ in 0..50 {
            last_loss = trainer.train_step(&input, &target);
        }
        assert!(
            last_loss < first_loss,
            "Loss should decrease after repeated training on same input: first={}, last={}",
            first_loss, last_loss
        );
    }

    #[test]
    fn test_train_epoch_returns_average_loss() {
        let (mut trainer, _dir) = setup_trainer();
        let avg_loss = trainer.train_epoch();
        assert!(avg_loss.is_finite(), "Average epoch loss should be finite, got {}", avg_loss);
        assert!(avg_loss > 0.0, "Average epoch loss should be positive, got {}", avg_loss);
    }
}
