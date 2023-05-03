use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../../gpu/gemm_gpu")
        .copy_to("../../resources/gemm.ptx")
        .build()
        .unwrap();
}
