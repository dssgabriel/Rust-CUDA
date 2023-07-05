use clap::Parser;
use cust::{
    function::{BlockSize, GridSize},
    prelude::*,
};
use nanorand::{Rng, WyRand};

use std::{
    error::Error,
    time::{Duration, Instant},
};

const REPETITIONS: usize = 31;

const M: usize = 4096;
const N: usize = M;
const K: usize = M;

const BS: usize = 32;
const WPT: usize = 8;

static DGEMM_PTX: &str = include_str!("../../../resources/dgemm.ptx");

#[derive(Clone, Copy, Debug, Parser, PartialEq)]
struct Args {
    #[clap(short, long, default_value_t = M)]
    m: usize,
    #[clap(short, long, default_value_t = N)]
    n: usize,
    #[clap(short, long, default_value_t = K)]
    k: usize,
    #[clap(short, long, default_value_t = REPETITIONS)]
    repetitions: usize,
    #[clap(short, long, default_value_t = false)]
    debug: bool,
}

#[allow(non_snake_case)]
fn dgemm_host(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    beta: f64,
    A: &[f64],
    B: &[f64],
    C: &mut [f64],
) {
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
    let args = Args::parse();

    // Initialize CUDA context, module and stream
    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(DGEMM_PTX, &[])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Get kernels from the PTX module
    let dgemm_naive = module.get_function("dgemm_naive")?;
    let dgemm_optim = module.get_function("dgemm_optim")?;

    // Initialize RNG from seed
    let mut rng = WyRand::new_seed(42);

    // Use small coefficients to avoid the matrices' contents from growing too much
    let alpha = 0.002;
    let beta = 0.001;

    // Initialize matrices randomly
    let mut A = vec![0.0; args.m * args.k];
    rng.fill(&mut A);
    let mut B = vec![0.0; args.k * args.n];
    rng.fill(&mut B);
    let mut C = vec![0.0; args.m * args.n];
    rng.fill(&mut C);

    // Create host result matrix
    let mut h_C = C.clone();

    // Create device matrices for naive DGEMM
    let d_A_naive = DeviceBuffer::from_slice(&A)?;
    let d_B_naive = DeviceBuffer::from_slice(&B)?;
    let d_C_naive = DeviceBuffer::from_slice(&C)?;

    // Create device matrices for optimized DGEMM
    let d_A_optim = DeviceBuffer::from_slice(&A)?;
    let d_B_optim = DeviceBuffer::from_slice(&B)?;
    let d_C_optim = DeviceBuffer::from_slice(&C)?;

    // Define the thread grid dimensions (same for both kernels)
    let grid = GridSize::xy((args.m / BS) as u32, (args.n / BS) as u32);

    // Define the thread blocks dimensions (custom for each GPU implementation)
    let blocks_naive = BlockSize::xy(BS as u32, BS as u32);
    let blocks_optim = BlockSize::xy(BS as u32, (BS / WPT) as u32);

    let mut res_C_naive: Vec<f64> = Vec::new();
    let mut durations_naive = Vec::with_capacity(REPETITIONS);
    // Benchmark the naive DGEMM GPU implementation
    for i in 0..REPETITIONS {
        let t = Instant::now();
        unsafe {
            launch!(
                dgemm_naive<<<grid, blocks_naive, 0, stream>>>(
                    args.m,
                    args.n,
                    args.k,
                    alpha,
                    beta,
                    d_A_naive.as_device_ptr(),
                    d_A_naive.len(),
                    d_B_naive.as_device_ptr(),
                    d_B_naive.len(),
                    d_C_naive.as_device_ptr(),
                )
            )?;
        }
        stream.synchronize()?;

        // Register duration
        durations_naive.push((t.elapsed()).as_secs_f64());

        // Store result after the first iteration
        if i == 0 {
            res_C_naive = d_C_naive.as_host_vec()?;
        }
    }

    let mean_naive = durations_naive.iter().sum::<f64>() / REPETITIONS as f64;
    let sdev_naive = (durations_naive
        .iter()
        .fold(0.0, |acc, d| acc + (d - mean_naive) * (d - mean_naive))
        / (REPETITIONS as f64 - 1.0))
        .sqrt();

    let mut res_C_optim: Vec<f64> = Vec::new();
    let mut durations_optim = Vec::with_capacity(REPETITIONS);

    // Benchmark the optim DGEMM GPU implementation
    for i in 0..REPETITIONS {
        let t = Instant::now();
        unsafe {
            launch!(
                dgemm_optim<<<grid, blocks_optim, 0, stream>>>(
                    args.m,
                    args.n,
                    args.k,
                    alpha,
                    beta,
                    d_A_optim.as_device_ptr(),
                    d_A_optim.len(),
                    d_B_optim.as_device_ptr(),
                    d_B_optim.len(),
                    d_C_optim.as_device_ptr(),
                )
            )?;
        }
        stream.synchronize()?;

        // Register duration
        durations_optim.push((t.elapsed()).as_secs_f64());

        // Store result after the first iteration
        if i == 0 {
            res_C_optim = d_C_optim.as_host_vec()?;
        }
    }

    let mean_optim = durations_optim.iter().sum::<f64>() / REPETITIONS as f64;
    let sdev_optim = (durations_optim
        .iter()
        .fold(0.0, |acc, d| acc + (d - mean_optim) * (d - mean_optim))
        / (REPETITIONS as f64 - 1.0))
        .sqrt();

    // Verify that results after one iteration are correct
    if args.debug == true {
        dgemm_host(args.m, args.n, args.k, alpha, beta, &A, &B, &mut h_C);

        for (idx, (h, (n, o))) in h_C
            .iter()
            .zip(res_C_optim.iter().zip(res_C_naive.iter()))
            .enumerate()
        {
            debug_assert!(
                (h - n).abs() < std::f64::EPSILON * (args.m * args.n * args.k) as f64,
                "Naive differs from host at index {idx}"
            );
            debug_assert!(
                (h - o).abs() < std::f64::EPSILON * (args.m * args.n * args.k) as f64,
                "Optimized differs from host at index {idx}"
            );
        }
    }

    println!(
        "\x1b[1m{:20}{:20}{:20}{:20}{}\x1b[0m",
        "Implementation",
        "Matrix dimensions",
        "Grid dimensions",
        "Block dimensions",
        "Average runtime"
    );
    println!(
        "{:20}{:20}{:20}{:20}{}",
        "naive",
        format!("{}x{}", args.m, args.n),
        format!("{}x{}", grid.x, grid.y),
        format!("{}x{}", blocks_naive.x, blocks_naive.y),
        format!(
            "{:?} ± {:?}",
            Duration::from_secs_f64(mean_naive),
            Duration::from_secs_f64(sdev_naive)
        )
    );
    println!(
        "{:20}{:20}{:20}{:20}{}",
        "optimized",
        format!("{}x{}", args.m, args.n),
        format!("{}x{}", grid.x, grid.y),
        format!("{}x{}", blocks_optim.x, blocks_optim.y),
        format!(
            "{:?} ± {:?}",
            Duration::from_secs_f64(mean_optim),
            Duration::from_secs_f64(sdev_optim)
        )
    );

    Ok(())
}
