#!/bin/bash

echo "Installing NVIDIA Container Toolkit (nvidia-docker2)..."

# Add NVIDIA package repositories:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install the NVIDIA Container Toolkit:
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

echo "NVIDIA Container Toolkit installation complete."
echo "You should now be able to run Docker containers with GPU support."
echo "Try running './build_and_run.sh' again."