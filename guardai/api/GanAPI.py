import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch
import os
from api_utils import save_models, load_models, generate_samples, classify_samples, list_available_models, load_data_from_directory
from models import Generator, Discriminator

class GanAPI:
    """
    API for training, loading, and using GAN models for prompt injection classification.

    Attributes:
        models_dir (str): Directory for model storage.
        data_dir (str): Directory containing training data.
        generator: Generator model.
        discriminator: Discriminator model.
        device: Device to run models on (CPU/GPU).
    
    Methods:
        save: Save the current generator and discriminator models.
        load: Load saved generator and discriminator models.
        train: Train GAN models for prompt injection classification.
        classify: Classify a prompt as benign or adversarial.
        generate: Generate adversarial prompt embeddings.
        list_models: List all available saved models.
    """
    def __init__(self, 
                 models_dir='../models', 
                 data_dir='../data/train_loaders'):
        """
        Initialize the GAN API.
        
        Args:
            models_dir (str): Directory for model storage.
            data_dir (str): Directory containing training data.

        Returns:
            None
        
        Raises:
            ValueError: If models_dir or data_dir is not a string.
        """
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.generator = None
        self.discriminator = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
    
    def save(self, model_name: str) -> bool:
        """
        Save the current generator and discriminator models.
        
        Args:
            model_name (str): Name to identify the saved model.
        
        Returns:
            bool: True if successful.

        Raises:
            ValueError: If no models are available to save.
        """
        if self.generator is None or self.discriminator is None:
            raise ValueError("No models to save. Train or load models first.")
        
        return save_models(self.generator, 
                           self.discriminator, 
                           model_name, 
                           self.models_dir)
    
    def load(self, model_name: str) -> tuple:
        """
        Load saved generator and discriminator models.
        
        Args:
            model_name (str): Name of the model to load.
        
        Returns:
            tuple: (generator, discriminator) - The loaded models.
        
        Raises:
            FileNotFoundError: If models with the given name are not found.
        """
        self.generator, self.discriminator = load_models(model_name, 
                                                         self.models_dir)
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        return self.generator, self.discriminator

    def train(self, 
              generator=None, 
              discriminator=None, 
              noise_dim=16, 
              hidden_dim=32, 
              output_dim=8, 
              batch_size=64, 
              num_epochs=10, 
              g_lr=0.0002, 
              d_lr=0.0002, 
              plot_data=True, 
              use_case=None, 
              smooth_factor=0.1
              ) -> tuple:
        """
        Trains the GAN models.
        
        Args:
            generator: Optional pre-initialized generator model.
            discriminator: Optional pre-initialized discriminator model.
            noise_dim (int): Input dimension for the generator.
            hidden_dim (int): Hidden dimension for both models.
            output_dim (int): Output dimension (prompt embedding size).
            batch_size (int): Training batch size.
            num_epochs (int): Number of training epochs.
            g_lr (float): Learning rate for the generator.
            d_lr (float): Learning rate for the discriminator.
            plot_data (bool): Plot the loss curves after training.
            use_case (str): Specific use case to train on (e.g., "malware", "intrusion").
                            If None, all available data will be used.
            smooth_factor (float): Label smoothing factor (0.0-0.5). Default 0.1 (means real=0.9, fake=0.1)
        
        Returns:
            tuple: (generator, discriminator) - The trained models.

        Raises:
            ValueError: If no training data is found.
        """
        def _plot_data(self, 
                       D_losses: list, 
                       G_losses: list
                       ) -> None:
            """
            Plot the loss curves for the discriminator and generator.

            Args:
                D_losses (list): List of discriminator losses.
                G_losses (list): List of generator losses.

            Returns:
                None
            
            Raises:
                ValueError: If d_losses and g_losses are not the same length.
            """
            plt.plot(D_losses, label='Discriminator Loss')
            plt.plot(G_losses, label='Generator Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

        if generator is None:
            generator = Generator(noise_dim, hidden_dim, output_dim)
        
        if discriminator is None:
            discriminator = Discriminator(output_dim, hidden_dim)
        
        self.generator = generator
        self.discriminator = discriminator
        
        generator.to(self.device)
        discriminator.to(self.device)
        
        dataloader = load_data_from_directory(self.data_dir, 
                                              batch_size, 
                                              use_case)
        
        criterion = nn.BCELoss()
        g_optimizer = optim.Adam(generator.parameters(), 
                                 lr=g_lr, 
                                 betas=(0.5, 0.999))
        d_optimizer = optim.Adam(discriminator.parameters(), 
                                 lr=d_lr, 
                                 betas=(0.5, 0.999))
        
        d_losses = []
        g_losses = []
        
        real_label_value = 1.0 - smooth_factor
        fake_label_value = smooth_factor
        
        for epoch in range(num_epochs):
            running_d_loss = 0.0
            running_g_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                real_samples = batch[0].to(self.device)
                batch_size = real_samples.size(0)
                
                d_optimizer.zero_grad()
                real_labels = torch.ones(batch_size, 1).to(self.device) * real_label_value
                d_real_output = discriminator(real_samples)
                d_real_loss = criterion(d_real_output, real_labels)
                
                noise = torch.randn(batch_size, noise_dim, device=self.device)
                fake_samples = generator(noise)
                fake_labels = torch.ones(batch_size, 1).to(self.device) * fake_label_value
                d_fake_output = discriminator(fake_samples.detach())
                d_fake_loss = criterion(d_fake_output, fake_labels)
                
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                d_optimizer.step()
                
                g_optimizer.zero_grad()
                noise = torch.randn(batch_size, noise_dim, device=self.device)
                fake_samples = generator(noise)
                fake_labels_for_g = torch.ones(batch_size, 1).to(self.device)
                d_output_for_g = discriminator(fake_samples)
                g_loss = criterion(d_output_for_g, fake_labels_for_g)
                g_loss.backward()
                g_optimizer.step()
                
                running_d_loss += d_loss.item()
                running_g_loss += g_loss.item()
                num_batches += 1
            
            epoch_d_loss = running_d_loss / num_batches
            epoch_g_loss = running_g_loss / num_batches
            d_losses.append(epoch_d_loss)
            g_losses.append(epoch_g_loss)
            
            print(f"Epoch [{epoch + 1}/{num_epochs}] | D_loss: {epoch_d_loss:.4f} | G_loss: {epoch_g_loss:.4f}")

        if plot_data:
            _plot_data(self, d_losses, g_losses)
        
        return self.generator, self.discriminator
    
    def classify(self, 
                 prompt_embedding: torch.Tensor, 
                 threshold=0.5
                 ) -> str:
        """
        Classify a prompt as benign or adversarial.
        
        Args:
            prompt_embedding (torch.Tensor): Embedded prompt to classify.
            threshold (float): Classification threshold.
        
        Returns:
            str: "benign" or "adversarial".

        Raises:
            ValueError: If no discriminator model is loaded.
            TypeError: If prompt_embedding is not a torch.Tensor.
        """
        if self.discriminator is None:
            raise ValueError("No discriminator model loaded. Call load() or train() first.")
        
        if not isinstance(prompt_embedding, torch.Tensor):
            raise TypeError("prompt_embedding must be a torch.Tensor")
        
        if len(prompt_embedding.shape) == 1:
            prompt_embedding = prompt_embedding.unsqueeze(0)
        
        results = classify_samples(self.discriminator, 
                                   prompt_embedding, 
                                   threshold, 
                                   self.device)
        
        if results['raw_scores'].mean().item() >= threshold:
            return "benign"
        else:
            return "adversarial"
    
    def generate(self, 
                 num_prompts=1
                 ) -> torch.Tensor:
        """
        Generate adversarial prompt embeddings.
        
        Args:
            num_prompts (int): Number of prompt embeddings to generate.
        
        Returns:
            torch.Tensor: Generated prompt embeddings.

        Raises:
            ValueError: If no generator model is loaded.
        """
        if self.generator is None:
            raise ValueError("No generator model loaded. Call load() or train() first.")
        
        first_layer = list(self.generator.model.children())[0]
        noise_dim = first_layer.in_features
        
        return generate_samples(self.generator, 
                                num_prompts, 
                                noise_dim, 
                                self.device)
    
    def list_models(self) -> list:
        """
        List all available saved models.
        
        Args:
            None

        Returns:
            list: Names of available models.
        
        Raises:
            FileNotFoundError: If no models are found.
        """
        return list_available_models(self.models_dir)