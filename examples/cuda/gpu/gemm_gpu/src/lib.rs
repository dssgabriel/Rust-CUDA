#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]

use cuda_std::prelude::*;
use thread::{block_dim_x, block_dim_y, block_idx_x, block_idx_y, thread_idx_x, thread_idx_y};

#[kernel]
#[allow(
    improper_ctypes_definitions,
    clippy::missing_safety_doc,
    non_snake_case
)]
pub unsafe fn gemm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
    A: &[f32],
    B: &[f32],
    C: *mut f32,
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
