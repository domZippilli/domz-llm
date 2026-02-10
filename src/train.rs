use std::{f64, usize};

use tch::{
    Device, Tensor,
    nn::{Adam, Optimizer, OptimizerConfig, VarStore},
};

use crate::{constants::VOCAB_SIZE, dataset::DataSet, model::MiniGPT};

pub struct MiniGPTTrainer {
    device: Device,
    var_store: VarStore,
    model: MiniGPT,
    optimizer: Optimizer,
    dataset: DataSet,
    max_batches: Option<usize>,
}

impl MiniGPTTrainer {
    pub fn new(device: Device, dataset: DataSet, max_batches: Option<usize>) -> MiniGPTTrainer {
        let var_store = VarStore::new(device);
        let model = MiniGPT::new(&var_store.root());
        let optimizer = Adam::default()
            .build(&var_store, 1e-4)
            .expect("failed to create Adam optimizer");
        MiniGPTTrainer {
            device,
            var_store,
            model,
            optimizer,
            dataset,
            max_batches,
        }
    }

    // Train the model for the given epochs.
    pub fn train(&mut self, epochs: usize) {
        std::fs::create_dir_all("checkpoints")
            .expect("failed to create checkpoints directory for training");
        for epoch in 0..epochs {
            println!("starting epoch: {}", epoch);
            let loss = self.train_epoch(epoch);
            println!("epoch {} loss: {}", epoch, loss);
            let path = format!("checkpoints/epoch_{}.safetensors", epoch);
            self.var_store
                .save(&path)
                .expect("failed to save checkpoint");
            println!("saved checkpoint to {}", path);
            std::fs::copy(&path, "checkpoints/latest.safetensors")
                .expect("could not copy checkpoint to latest");
            println!("saved checkpoint to checkpoints/latest.safetensors");
        }
    }

    // Perform one epoch.
    fn train_epoch(&mut self, epoch: usize) -> f64 {
        // Do some averaging of the loss over each input
        let mut total_loss = 0.0;
        let mut inputs = 0;
        let training_data = self.dataset.get_training_dataset();
        for (i, batch) in training_data.into_iter().enumerate() {
            if let Some(max_batches) = self.max_batches {
                if i >= max_batches {
                    println!(
                        "ending epoch {} at batch: {} due to maximum batch threshold",
                        epoch, i
                    );
                    break;
                }
            }
            println!("starting epoch {} batch {}", epoch, i);
            // Move the batch to the device and capture the loss for book-keeping.
            for (j, (input, target)) in batch.iter().enumerate() {
                inputs += 1;
                let loss = self.train_step(
                    &input.to_device(self.device),
                    &target.to_device(self.device),
                );
                total_loss += loss;
                if j % 10 == 0 {
                    println!(
                        "step {}.{}.{} loss: {:.6}, epoch avg loss: {:.6}",
                        epoch,
                        i,
                        j,
                        loss,
                        total_loss / inputs as f64
                    )
                }
            }
            println!("batch {} avg loss: {:.6}", i, total_loss / inputs as f64)
        }
        total_loss / inputs as f64
    }

    // Perform one training step.
    fn train_step(&mut self, input: &Tensor, target: &Tensor) -> f64 {
        // Scan for out-of-range inputs
        // let max_val = i64::try_from(&input.max()).unwrap();
        // let min_val = i64::try_from(&input.min()).unwrap();
        // println!(
        //     "input range: [{}, {}], VOCAB_SIZE: {}",
        //     min_val, max_val, VOCAB_SIZE
        // );

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
        let trainer = MiniGPTTrainer::new(Device::Cpu, dataset, None);
        (trainer, dir) // keep dir alive so temp files aren't deleted
    }

    #[test]
    fn test_train_step_loss_is_finite() {
        let (mut trainer, _dir) = setup_trainer();
        // Create a simple synthetic input/target pair
        let input = Tensor::from_slice(&[100i64, 101, 102, 32, 102, 111, 111, 40]);
        let target = Tensor::from_slice(&[101i64, 102, 32, 102, 111, 111, 40, 41]);
        let loss = trainer.train_step(&input, &target);
        assert!(
            loss.is_finite(),
            "Loss should be a finite number, got {}",
            loss
        );
        assert!(
            loss > 0.0,
            "Cross-entropy loss should be positive, got {}",
            loss
        );
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
            first_loss,
            last_loss
        );
    }

    #[test]
    fn test_train_epoch_returns_average_loss() {
        let (mut trainer, _dir) = setup_trainer();
        let avg_loss = trainer.train_epoch(0);
        assert!(
            avg_loss.is_finite(),
            "Average epoch loss should be finite, got {}",
            avg_loss
        );
        assert!(
            avg_loss > 0.0,
            "Average epoch loss should be positive, got {}",
            avg_loss
        );
    }
}
