#!/bin/bash

set -e

# Check if dataloaders exist in the data directory:
if [ $(find ./data -name "train_loader_chunk_*" | wc -l) -eq 0 ]; then
    echo "Error: Chunked dataloaders not found in ./data directory."
    echo "Please ensure files named 'train_loader_chunk_#' exist in the data directory."
    exit 1
fi

# Install GAN API inside the container:
cd ./api
pip install .
cd ..

echo "Dataloaders found. Starting GAN training..."
python3 ./api/example.py

# Keep container running for debugging if needed:
# tail -f /dev/null