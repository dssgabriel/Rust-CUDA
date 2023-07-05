# DGEMM benchmark example

Simple performance comparison of two implementation of a DGEMM in Rust-CUDA:
- a naive one with no particular optimization;
- a tiled one which leverages shared memory and attributes more work per device thread to force vectorized loads and stores.
