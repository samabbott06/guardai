#!/bin/bash

# Get the absolute path of the directory containing this script.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

sudo docker build -t gan-sim .

# Run nvidia-docker for GPU support in container:
if sudo docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
    echo "Running container with GPU support..."
    
    sudo docker run -it --rm --gpus all \
      -v "$PROJECT_ROOT":/app \
      gan-sim
else
    echo "WARNING: NVIDIA Docker not available. Falling back to CPU mode."
    echo "If you want to use GPU acceleration, please run: ./install_nvidia_docker.sh"
    echo "Running without GPU support..."

    sudo docker run -it --rm \
      -v "$PROJECT_ROOT":/app \
      gan-sim
fi