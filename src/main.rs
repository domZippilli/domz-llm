use libc::dlopen;
use tch;

fn main() {
    unsafe {
        libc::dlopen(
            c"/usr/local/lib/libtorch/lib/libtorch_cuda.so".as_ptr(),
            libc::RTLD_LAZY,
        );
    }
    assert!(tch::Cuda::is_available());
    println!("Hello, world!");
}
