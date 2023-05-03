FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Update default packages
RUN apt-get update

# Install the necessary libraries and tools
RUN apt-get install -y build-essential curl xz-utils pkg-config libssl-dev zlib1g-dev libtinfo-dev libxml2-dev libncurses5 vim

# Update and upgrade the new packages
RUN apt-get update && apt-get upgrade -y

# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
# Download the appropriate `nightly` profile and the required components
# This avoids redownloading the correct `rustc` version everytime one runs from an already built image
RUN /root/.cargo/bin/rustup toolchain install nightly-2021-12-04 --component rust-src rustc-dev llvm-tools-preview

# Get prebuilt LLVM 7.0.1
RUN curl -O https://releases.llvm.org/7.0.1/clang+llvm-7.0.1-x86_64-linux-gnu-ubuntu-18.04.tar.xz && \
    xz -d /clang+llvm-7.0.1-x86_64-linux-gnu-ubuntu-18.04.tar.xz && \
    tar xf /clang+llvm-7.0.1-x86_64-linux-gnu-ubuntu-18.04.tar && \
    mv /clang+llvm-7.0.1-x86_64-linux-gnu-ubuntu-18.04 /root/llvm && \
    rm /clang+llvm-7.0.1-x86_64-linux-gnu-ubuntu-18.04.tar

# Set the appropriate environment variables
ENV RUST_LOG=info
ENV CARGO_HOME=/root/.cargo
ENV CUDA_HOME=/usr/local/cuda
ENV LLVM_HOME=/root/llvm
ENV LLVM_CONFIG=$LLVM_HOME/bin/llvm-config
ENV LLVM_LINK_STATIC=1

ENV PATH=$LLVM_HOME/bin:$CUDA_HOME/bin:$CUDA_HOME/nvvm/bin:$CARGO_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$LLVM_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64:$LD_LIBRARY_PATH
ENV CPATH=$LLVM_HOME/include:$CUDA_HOME/include:$CUDA_HOME/nvvm/include:$CPATH

# Make `ld` aware of the necessary dynamic/shared libraries
RUN echo $LLVM_HOME/lib >> /etc/ld.so.conf && \
    echo $CUDA_HOME/lib64 >> /etc/ld.so.conf && \
    echo $CUDA_HOME/nvvm/lib64 >> /etc/ld.so.conf && \
    echo $CUDA_HOME/compat >> /etc/ld.so.conf && \
    ldconfig
