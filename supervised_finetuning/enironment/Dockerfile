# Base image with PyTorch + CUDA
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Set working directory
WORKDIR /workspace

# Install minimal system tools
RUN apt-get update && apt-get install -y \
    wget git tmux bzip2 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Optional: print conda version
RUN conda --version

# Default to bash shell
CMD ["/bin/bash"]
