import torch.nn as nn
import torch

class Generator(nn.Module):
    """Generator Model Definition"""
    def __init__(self, 
                 noise_dim: int, 
                 hidden_dim: int, 
                 output_dim: int):
        """
        Generator Network initialization.

        Args:
            noise_dim (int): Dimension of the input noise vector.
            output_dim (int): Dimension of the generated output (matches the real data size).
            hidden_dim (int): Number of hidden units in each hidden layer.
        
        Returns:
            torch.Tensor: Generated data.

        Raises:
            ValueError: If noise_dim or output_dim is less than 1.
        """
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, output_dim),
            nn.Tanh()
        )

    def forward(self, 
                noise: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass of the generator network.

        Args:
            noise (torch.Tensor): Random noise tensor.

        Returns:
            torch.Tensor: Generated data.

        Raises:
            ValueError: If noise is not a 2D tensor.
        """
        return self.model(noise)

class Discriminator(nn.Module):
    """Discriminator Model Definition"""
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int):
        """
        Discriminator Network initialization.

        Args:
            input_dim (int): Dimension (matches real data size).
            hidden_dim (int): Number of hidden units in each hidden layer.

        Returns:
            torch.Tensor: Output probability that input is real.

        Raises:
            ValueError: If input_dim is less than 1.
        """
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, 
                data: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass of the discriminator network.

        Args:
            data (torch.Tensor): Real or generated data.

        Returns:
            torch.Tensor: Output probability that input is real.
        
        Raises:
            ValueError: If data is not a 2D tensor.
        """
        return self.model(data)