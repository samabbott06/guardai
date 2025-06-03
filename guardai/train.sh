#!/bin/bash

# Use this for training new GAN models.
# Make sure your working directory is set to the project root.

python3 api/example.py \
  --model-name "manipulated_content" \
  --data-dir "/home/nickh/Desktop/School/CSCI_493/2025-ai-attack-surfaces/data/train_loaders/manipulated_content" \
  --models-dir "models/manipulated_content" \
  --generator-name "manipulated_content_generator" \
  --discriminator-name "manipulated_content_discriminator" \
  --epochs 100 \
  --g-lr 0.0004 \
  --d-lr 0.0001 \
  --smooth-factor 0.1