#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]

use cuda_std::{prelude::*, shared_array};
use thread::{
    block_dim_x, block_dim_y, block_idx_x, block_idx_y, sync_threads, thread_idx_x, thread_idx_y,
};

/// The thread block size. This assumes that the user launches the kernel with thread blocks of
/// dimensions 32x32.
const BS: usize = 32;

/// Work per thread.
const WPT: usize = 8;
const RBS: usize = BS / WPT;

#[kernel]
#[allow(
    improper_ctypes_definitions,
    clippy::missing_safety_doc,
    non_snake_case
)]
pub unsafe fn dgemm_naive(
    m: usize,
    _n: usize,
    k: usize,
    alpha: f64,
    beta: f64,
    A: &[f64],
    B: &[f64],
    C: *mut f64,
) {
    let i = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;
    let j = (block_idx_y() * block_dim_y() + thread_idx_y()) as usize;

    let mut acc = 0.0;
    for l in 0..k {
        acc += A[i + m * l] * B[l + k * j];
    }

    let item = &mut *C.add(i + m * j);
    *item *= beta + acc * alpha;
}

#[kernel]
#[allow(
    improper_ctypes_definitions,
    clippy::missing_safety_doc,
    non_snake_case
)]
pub unsafe fn dgemm_optim(
    m: usize,
    _n: usize,
    k: usize,
    alpha: f64,
    beta: f64,
    A: &[f64],
    B: &[f64],
    C: *mut f64,
) {
    let lda = m;
    let ldb = k;
    let ldc = m;

    let tidx = thread_idx_x() as usize;
    let tidy = thread_idx_y() as usize;
    let gidx = block_idx_x() as usize * BS + tidx;
    let gidy = block_idx_y() as usize * BS + tidy;

    let Ab = shared_array![f64; BS * BS];
    let Bb = shared_array![f64; BS * BS];

    let mut acc = [0.0_f64; WPT];

    for b in 0..(k / BS) {
        let bidx = b * BS + tidx;
        let bidy = b * BS + tidy;
        for w in 0..WPT {
            *Ab.add((tidy + w * RBS) * BS + tidx) = A[(bidy + w * RBS) * lda + gidx];
            *Bb.add((tidy + w * RBS) * BS + tidx) = B[(gidy + w * RBS) * ldb + bidx];
        }
        sync_threads();

        for l in 0..BS {
            for (w, item) in acc.iter_mut().enumerate().take(WPT) {
                *item += *Ab.add(l * BS + tidx) * *Bb.add((tidy + w * RBS) * BS + l);
            }
        }
        sync_threads();
    }

    for (w, item) in acc.iter().enumerate().take(WPT) {
        *C.add((gidy + w * RBS) * ldc + gidx) *= beta + alpha * item;
    }
}
