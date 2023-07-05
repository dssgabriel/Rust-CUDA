use cuda_builder::CudaBuilder;

fn main() {
    println!("cargo:rerun-if-changed=../../resources/dgemm.ptx");
    CudaBuilder::new("../../gpu/bench_dgemm_gpu")
        .arch(cuda_builder::NvvmArch::Compute75)
        .copy_to("../../resources/dgemm.ptx")
        .build()
        .unwrap();
}
