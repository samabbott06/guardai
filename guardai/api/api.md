# General Adversarial Network (GAN) API

## Overview
A bespoke Python API for loading, saving, training, creating, and interacting with Generative Adversarial Networks (GANs) specifically designed for prompt injection classification. This API provides tools to detect and generate adversarial prompts for AI systems.

## Features
- **Training**: Train GAN models on different datasets with customizable parameters
- **Classification**: Classify prompt embeddings as benign or adversarial
- **Generation**: Generate adversarial prompt embeddings
- **Model Management**: Save, load, and list available models
- **Use Case Specific Models**: Support for different attack surfaces (malware, intrusion, etc.)

## Installation
- Change your current working directory to `/api`
- Run `pip install .` to install `GanAPI`
- Python 3.10.12 or higher is required

## Directory Structure
```
api/
  ├── GanAPI.py        # Main API implementation
  ├── models.py        # Generator and Discriminator model definitions
  ├── api_utils.py     # Helper functions for the API
  ├── __init__.py      # Package initialization
  └── setup.py         # Package setup configuration
```

## Basic Usage

```python
from GanAPI import GanAPI

# Initialize the API
api = GanAPI(models_dir='models', data_dir='data/train_loaders')

# Train a model
api.train(noise_dim=16, hidden_dim=32, output_dim=8, num_epochs=10, use_case="malware")

# Save the trained model
api.save("malware_model")

# Load an existing model
api.load("malware_model")

# Generate adversarial prompt embeddings
generated_embeddings = api.generate(num_prompts=5)

# Classify a prompt embedding
result = api.classify(prompt_embedding, threshold=0.5)  # Returns "benign" or "adversarial"

# List available models
available_models = api.list_models()
```

## API Reference

### GanAPI Class

#### Constructor
```python
GanAPI(models_dir='../models', data_dir='../data/train_loaders')
```

#### Methods

##### `save(model_name: str) -> bool`
Save the current generator and discriminator models.

##### `load(model_name: str) -> tuple`
Load saved generator and discriminator models.

##### `train(generator=None, discriminator=None, noise_dim=16, hidden_dim=32, output_dim=8, batch_size=64, num_epochs=10, g_lr=0.0002, d_lr=0.0002, plot_data=True, use_case=None, smooth_factor=0.1) -> tuple`
Train GAN models for prompt injection classification.

##### `classify(prompt_embedding: torch.Tensor, threshold=0.5) -> str`
Classify a prompt as benign or adversarial.

##### `generate(num_prompts=1) -> torch.Tensor`
Generate adversarial prompt embeddings.

##### `list_models() -> list`
List all available saved models.

## Use Cases
The API supports different use cases for training and classification:
- `info_gathering`: Information gathering attacks
- `intrusion`: Intrusion attempts
- `malware`: Malware-related attacks
- `manipulated_content`: Content manipulation attacks

## Updating
In the event that the API requires any changes or is added to, follow these steps:

> **NOTE**: Please maintain the version number in `__init__.py` and `setup.py` when making any changes to the API. Current version: 0.1.0

- Change your current working directory to `/api`
- Run `pip install . --force-reinstall` to update the API installation

## Requirements
- Python 3.10.12 or higher
- PyTorch
- Matplotlib (for visualization)