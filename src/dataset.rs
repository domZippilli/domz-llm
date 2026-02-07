use rand;
use tch::Tensor;
use walkdir;

use crate::tokenizer;

const CONTEXT_LENGTH: usize = 512;
const TRAIN_VALIDATION_SPLIT: f32 = 0.9;

#[derive(Debug, Clone)]
pub struct DataSet {
    training: DataSubset,
    validation: DataSubset,
}

impl DataSet {
    pub fn new(data_dir: String, tokenizer: tokenizer::Tokenizer) -> Self {
        // Enumerate all files in the data directory and split them into training and validation sets.
        let file_paths = Self::enumerate_files(&data_dir);
        let mut training_paths =
            Vec::with_capacity(file_paths.len() * TRAIN_VALIDATION_SPLIT as usize);
        let mut validation_paths = Vec::with_capacity(file_paths.len() - training_paths.len());
        for path in &file_paths {
            if rand::random::<f32>() < TRAIN_VALIDATION_SPLIT {
                training_paths.push(path.clone());
            } else {
                validation_paths.push(path.clone());
            }
        }
        DataSet {
            training: DataSubset::new(training_paths, tokenizer.clone()),
            validation: DataSubset::new(validation_paths, tokenizer.clone()),
        }
    }

    fn enumerate_files(data_dir: &str) -> Vec<String> {
        walkdir::WalkDir::new(data_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.path().to_string_lossy().to_string())
            .collect()
    }

    pub fn get_training_dataset(&self) -> DataSubset {
        self.training.clone()
    }

    pub fn get_validation_dataset(&self) -> DataSubset {
        self.validation.clone()
    }
}

#[derive(Debug, Clone)]
pub struct DataSubset {
    paths: Vec<String>,
    tokenizer: tokenizer::Tokenizer,
}

impl DataSubset {
    pub fn new(paths: Vec<String>, tokenizer: tokenizer::Tokenizer) -> Self {
        DataSubset { paths, tokenizer }
    }
}

impl Iterator for DataSubset {
    type Item = Vec<(Tensor, Tensor)>; // (input, target) batch data

    /// Returns the next batch of input-target pairs for the dataset, chunked.
    fn next(&mut self) -> Option<Self::Item> {
        // Load the next batch of data from the remaining dataset files.
        if let Some(batch_file) = self.paths.pop() {
            let text = std::fs::read_to_string(batch_file).expect("Failed to read file");
            let mut tokens = self.tokenizer.encode(&text).expect("Failed to encode text");
            tokens.push(tokenizer::EOS_TOKEN); // Append EOS token
            let chunks: Vec<Vec<u8>> = tokens
                .chunks(CONTEXT_LENGTH)
                .map(|token_chunk| Vec::from(token_chunk))
                .collect();
            let mut batch: Vec<(Tensor, Tensor)> = Vec::new();
            let final_chunk = chunks.len() - 1;
            for (i, chunk) in chunks.into_iter().enumerate() {
                let mut input_vec = chunk[..chunk.len() - 1].to_vec();
                let mut target_vec = chunk[1..].to_vec();
                if i == final_chunk {
                    // Pad the last chunk
                    input_vec.resize(CONTEXT_LENGTH - 1, tokenizer::PAD_TOKEN);
                    target_vec.resize(CONTEXT_LENGTH - 1, tokenizer::PAD_TOKEN);
                }
                if input_vec.len() < 2 {
                    continue; // Skip chunks that are too small
                }
                let input_tensor = Tensor::from_slice(&input_vec).to_kind(tch::Kind::Int64);
                let target_tensor = Tensor::from_slice(&target_vec).to_kind(tch::Kind::Int64);
                batch.push((input_tensor, target_tensor));
            }
            Some(batch)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn setup_test_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..10 {
            let path = dir.path().join(format!("test_{}.py", i));
            let mut f = std::fs::File::create(path).unwrap();
            writeln!(f, "def foo_{}():\n    return {}", i, i).unwrap();
        }
        dir
    }

    #[test]
    fn test_train_val_split() {
        let dir = setup_test_dir();
        let ds = DataSet::new(dir.path().to_string_lossy().to_string(), tokenizer::Tokenizer::new());
        let train_count = ds.get_training_dataset().paths.len();
        let val_count = ds.get_validation_dataset().paths.len();
        assert_eq!(train_count + val_count, 10);
        assert!(train_count > 0, "training set should not be empty");
    }

    #[test]
    fn test_chunk_sizes_consistent() {
        let dir = setup_test_dir();
        let ds = DataSet::new(dir.path().to_string_lossy().to_string(), tokenizer::Tokenizer::new());
        let mut training = ds.get_training_dataset();
        if let Some(batch) = training.next() {
            for (input, target) in &batch {
                let input_size = input.size();
                let target_size = target.size();
                assert_eq!(input_size, target_size, "input and target should be same length");
                assert_eq!(input_size[0], (CONTEXT_LENGTH - 1) as i64, "chunks should be CONTEXT_LENGTH - 1");
            }
        }
    }

    #[test]
    fn test_epoch_via_clone() {
        let dir = setup_test_dir();
        let ds = DataSet::new(dir.path().to_string_lossy().to_string(), tokenizer::Tokenizer::new());
        let epoch1: Vec<_> = ds.get_training_dataset().collect();
        let epoch2: Vec<_> = ds.get_training_dataset().collect();
        assert_eq!(epoch1.len(), epoch2.len(), "cloned epochs should yield same number of batches");
    }

    #[test]
    fn test_iterator_exhausts() {
        let dir = setup_test_dir();
        let ds = DataSet::new(dir.path().to_string_lossy().to_string(), tokenizer::Tokenizer::new());
        let mut training = ds.get_training_dataset();
        let count = training.by_ref().count();
        assert!(count > 0);
        assert!(training.next().is_none(), "should be exhausted after full iteration");
    }
}


