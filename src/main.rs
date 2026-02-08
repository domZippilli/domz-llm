#[allow(unused_imports)]
use libc::dlopen;
use tch;

mod constants;
mod dataset;
mod tokenizer;
mod model;
mod train;

fn main() {
    // INITIALIZATION
    // Kludge to load CUDA since libtorch seems to load it lazily, causing is_available to return false.
    unsafe {
        libc::dlopen(
            c"/usr/local/lib/libtorch/lib/libtorch_cuda.so".as_ptr(),
            libc::RTLD_LAZY,
        );
    }
    assert!(tch::Cuda::is_available());
    // Load tokenizer to ensure it compiles
    let tokenizer = tokenizer::Tokenizer::new();
    let dataset = dataset::DataSet::new("data/".to_string(), tokenizer);
    let mut trainer = train::MiniGPTTrainer::new(tch::Device::Cuda(0), dataset, Some(300));
    trainer.train(5);
}
