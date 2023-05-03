use cust::{
    function::{BlockSize, GridSize},
    prelude::*,
};
use nanorand::{Rng, WyRand};

use std::error::Error;

/// How many numbers to generate and add together.
const MATRIX_SIZE: usize = 2048;
const BLOCK_SIZE: u32 = 32;

const ALPHA: f32 = 0.2;
const BETA: f32 = 0.1;

static PTX: &str = include_str!("../../../resources/gemm.ptx");

#[allow(dead_code, non_snake_case)]
fn gemm(m: usize, n: usize, k: usize, alpha: f32, beta: f32, A: &[f32], B: &[f32], C: &mut [f32]) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for l in 0..k {
                acc += A[i + m * l] * B[l + k * j]
            }
            C[i + m * j] *= beta + alpha * acc;
        }
    }
}

#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn Error>> {
    // Generate our random vectors.
    let mut wyrand = WyRand::new();
    let mut h_A = vec![0.2_f32; MATRIX_SIZE * MATRIX_SIZE];
    wyrand.fill(&mut h_A);
    let mut h_B = vec![0.2_f32; MATRIX_SIZE * MATRIX_SIZE];
    wyrand.fill(&mut h_B);
    let mut h_C = vec![0.1_f32; MATRIX_SIZE * MATRIX_SIZE];
    wyrand.fill(&mut h_C);

    // Initialize CUDA, this will pick the first available device and will make a CUDA context from
    // it. We don't need the context for anything but it must be kept alive.
    let _ctx = cust::quick_init()?;

    // Make the CUDA module, modules just house the GPU code for the kernels we created.
    // They can be made from PTX code, cubins, or fatbins.
    let module = Module::from_ptx(PTX, &[])?;

    // Make a CUDA stream to issue calls to. You can think of this as an OS thread but for
    // dispatching GPU calls.
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Allocate the GPU memory needed to house our numbers and copy them over.
    let d_A = DeviceBuffer::from_slice(&h_A)?;
    let d_B = DeviceBuffer::from_slice(&h_B)?;
    let d_C = DeviceBuffer::from_slice(&h_C)?;

    // Retrieve the `gemm` kernel from the module.
    let func = module.get_function("gemm")?;
    let block_size = BlockSize::xy(BLOCK_SIZE, BLOCK_SIZE);
    let grid_size = GridSize::xy(
        MATRIX_SIZE as u32 / block_size.x,
        MATRIX_SIZE as u32 / block_size.y,
    );

    println!(
        "Using {:?} blocks and {:?} threads per block",
        grid_size, block_size
    );

    let now = std::time::Instant::now();
    for _ in 0..3 {
        // Actually launch the kernel on the device. This will queue up the launch on the stream without
        // blocking the main host thread.
        unsafe {
            launch!(
                // Slices are passed as two parameters, the pointer and the length.
                func<<<grid_size, block_size, 0, stream>>>(
                    MATRIX_SIZE,         // lda
                    MATRIX_SIZE,         // ldb
                    MATRIX_SIZE,         // ldc
                    ALPHA,               // alpha
                    BETA,                // beta
                    d_A.as_device_ptr(), // A
                    d_A.len(),           // A.len
                    d_B.as_device_ptr(), // B
                    d_B.len(),           // B.len
                    d_C.as_device_ptr(), // C
                )
            )?;
        }
    }
    stream.synchronize()?;
    let duration = now.elapsed() / 3;

    // Copy back the data from the device.
    let res = d_C.as_host_vec()?;

    println!(
        "Matrix size: {MATRIX_SIZE}, C[0] = {}\nDone in {duration:?}",
        res[0]
    );

    Ok(())
}
