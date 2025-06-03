import torch
import glob
import os
from torch.utils.data import TensorDataset, ConcatDataset
from models import Generator, Discriminator
from torch.utils.data import DataLoader

def save_models(generator: Generator, 
                discriminator: Discriminator, 
                model_name: str, 
                models_dir: str
                ) -> bool:
    """
    Save generator and discriminator models.
    
    Args:
        generator: Generator model.
        discriminator: Discriminator model.
        model_name (str): Name to identify the saved model.
        models_dir (str): Directory for model storage.
    
    Returns:
        bool: True if successful.
    
    Raises:
        OSError: If exist_ok is False and the directory already exists.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    gen_path = os.path.join(models_dir, f"{model_name}_generator.pt")
    disc_path = os.path.join(models_dir, f"{model_name}_discriminator.pt")
    
    torch.save(generator.state_dict(), gen_path)
    torch.save(discriminator.state_dict(), disc_path)
    
    print(f"Models saved to {gen_path} and {disc_path}")

    return True

def load_data_from_directory(data_dir='data/train_loaders', 
                             batch_size=64, 
                             use_case=None
                             ) -> DataLoader:
    """
    Load data from directory containing .pt files.

    Args:
        data_dir (str): Directory path containing .pt files.
        batch_size (int): DataLoader batch size.
        use_case (str): Specific use case subdirectory to load data from.

    Returns:
        DataLoader: DataLoader containing the loaded datasets.

    Raises:
        FileNotFoundError: If data_dir is invalid or no .pt files are found.
        ValueError: If no valid datasets could be loaded.
    """
    # If use_case is specified, update the directory path.
    if use_case:
        data_dir = os.path.join(data_dir, use_case)
        print(f"Loading data for use case: {use_case} from {data_dir}")
    
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    datasets = []
    print(f"Loading dataloaders from directory: {data_dir}")
    
    # Load all .pt files directly from the specified directory.
    pt_files = glob.glob(os.path.join(data_dir, "*.pt"))
    
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")
    
    print(f"Found {len(pt_files)} .pt files in {data_dir}")
    
    for file in pt_files:
        try:
            dataset = _extract_dataset_from_file(file)
            if dataset:
                datasets.append(dataset)
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    
    if not datasets:
        raise ValueError("No valid datasets could be loaded")
    
    combined_dataset = ConcatDataset(datasets)
    
    return DataLoader(combined_dataset,
                      batch_size=batch_size,
                      shuffle=True)

def _extract_dataset_from_file(file_path: str) -> TensorDataset:
    """
    Helper function to extract dataset from a .pt file
    
    Args:
        file_path (str): Path to the .pt file
        
    Returns:
        TensorDataset or None: The extracted dataset or None if extraction failed
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        loaded_data = torch.load(file_path, 
                                 map_location=device, 
                                 weights_only=False)
        
        if hasattr(loaded_data, 'dataset'):
            if hasattr(loaded_data.dataset, 'tensors'):
                return TensorDataset(loaded_data.dataset.tensors[0])
            else:
                data_tensor = torch.stack([loaded_data.dataset[i][0] 
                                           for i in range(len(loaded_data.dataset))])
                return TensorDataset(data_tensor)
        else:
            print(f"Warning: Could not extract dataset from {file_path}. Skipping.")
            return None
    
    except Exception as e:
        print(f"Error extracting dataset from {file_path}: {str(e)}")
        return None

def load_models(model_name: str, 
                models_dir: str, 
                noise_dim=16, 
                hidden_dim=32, 
                output_dim=8
                ) -> tuple:
    """
    Load saved generator and discriminator models.
    
    Args:
        model_name (str): Name of the model to load.
        models_dir (str): Directory containing saved models.
        noise_dim, hidden_dim, output_dim: Model parameters if needed for recreation.
    
    Returns:
        tuple: (generator, discriminator) - The loaded models.

    Raises:
        FileNotFoundError: If models with the given name are not found.
    """
    gen_path = os.path.join(models_dir, f"{model_name}_generator.pt")
    disc_path = os.path.join(models_dir, f"{model_name}_discriminator.pt")
    
    if not os.path.exists(gen_path) or not os.path.exists(disc_path):
        raise FileNotFoundError(f"Models with name '{model_name}' not found in {models_dir}")
    
    generator = Generator(noise_dim, hidden_dim, output_dim)
    discriminator = Discriminator(output_dim, hidden_dim)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.load_state_dict(
        torch.load(gen_path, 
                   map_location=device, 
                   weights_only=False)
    )
    discriminator.load_state_dict(
        torch.load(disc_path, 
                   map_location=device, 
                   weights_only=False)
    )

    generator.eval()
    discriminator.eval()
    
    print(f"Models loaded from:\n{gen_path}\n{disc_path}")
    return generator, discriminator

def generate_samples(generator: Generator, 
                     num_samples: int,
                     noise_dim: int, 
                     device=None
                     ) -> torch.Tensor:
    """
    Generate samples using the generator model.
    
    Args:
        generator: Generator model.
        num_samples (int): Number of samples to generate.
        noise_dim (int): Noise vector dimension.
        device: Device to run generation on (CPU/GPU).
    
    Returns:
        torch.Tensor: Generated samples

    Raises:
        None
    """
    generator.eval()
    
    noise = torch.randn(num_samples, 
                        noise_dim, 
                        device=device)
    
    with torch.no_grad():
        samples = generator(noise)
    
    return samples

def classify_samples(discriminator: Discriminator, 
                     samples: torch.Tensor, 
                     threshold=0.5, 
                     device=None
                     ) -> dict:
    """
    Classify samples as real (benign) or fake (adversarial).
    
    Args:
        discriminator: Discriminator model.
        samples (torch.Tensor): Samples to classify.
        threshold (float): Classification threshold.
        device: Device to run classification on.
    
    Returns:
        dict: Classification results.

    Raises:
        None
    """
    discriminator.eval()
    
    if device is not None:
        samples = samples.to(device)
    
    with torch.no_grad():
        scores = discriminator(samples)
    
    predictions = (scores >= threshold).float()
    classification = ["benign" if p.item() 
                      else "adversarial" 
                      for p in predictions]
    
    return {'raw_scores': scores,
            'predictions': predictions,
            'classification': classification}

def list_available_models(models_dir: str) -> list:
    """
    List all available saved models.
    
    Args:
        models_dir (str): Directory containing saved models.
    
    Returns:
        list: Names of available models.

    Raises:
        None
    """
    if not os.path.exists(models_dir):
        return []
    
    gen_pattern = os.path.join(models_dir, "*_generator.pt")
    gen_files = glob.glob(gen_pattern)
    
    model_names = [os.path.basename(f).replace("_generator.pt", "") 
                  for f in gen_files]
    
    return model_names
