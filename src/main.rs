#[allow(unused_imports)]
use libc::dlopen;

use std::io::{self, Write};
use clap::{Parser, Subcommand};

mod constants;
mod dataset;
mod generate;
mod model;
mod tokenizer;
mod train;

const TRAINING_BATCHES: usize = 1000;
const TRAINING_EPOCHS: usize = 1000;

#[derive(Parser)]
#[command(name = "domz-llm", about = "GPT-style language model in Rust")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train the model
    Train {
        /// Path to training data directory
        #[arg(short, long, default_value = "data/")]
        data: String,
    },
    /// Generate text from a trained model
    Generate {
        /// Path to model checkpoint
        #[arg(short, long, default_value = "checkpoints/latest.safetensors")]
        checkpoint: String,
    },
}

fn init_cuda() {
    // Kludge to load CUDA since libtorch seems to load it lazily, causing is_available to return false.
    unsafe {
        libc::dlopen(
            c"/usr/local/lib/libtorch/lib/libtorch_cuda.so".as_ptr(),
            libc::RTLD_LAZY,
        );
    }
    assert!(tch::Cuda::is_available(), "CUDA is not available");
}

fn main() {
    init_cuda();
    let cli = Cli::parse();

    match cli.command {
        Command::Train { data } => {
            eprint!("This will overwrite existing checkpoints. Are you sure? [y/N] ");
            io::stderr().flush().unwrap();
            let mut answer = String::new();
            io::stdin().read_line(&mut answer).unwrap();
            if !answer.trim().eq_ignore_ascii_case("y") {
                eprintln!("Aborted.");
                return;
            }
            let tokenizer = tokenizer::Tokenizer::new();
            let dataset = dataset::DataSet::new(data, tokenizer);
            let mut trainer = train::MiniGPTTrainer::new(tch::Device::Cuda(0), dataset, Some(TRAINING_BATCHES));
            trainer.train(TRAINING_EPOCHS);
        }
        Command::Generate { checkpoint } => {
            let checkpoint_path = std::path::Path::new(&checkpoint);
            let generator = generate::MiniGPTGenerator::new(tch::Device::Cuda(0), checkpoint_path);

            loop {
                eprint!("? ");
                io::stderr().flush().unwrap();
                let mut prompt = String::new();
                io::stdin().read_line(&mut prompt).unwrap();
                let prompt = prompt.trim_end_matches('\n');
                if prompt.is_empty() {
                    break;
                }
                for c in generator.prompt(prompt) {
                    print!("{}", c);
                    io::stdout().flush().unwrap();
                }
                println!();
            }
        }
    }
}
