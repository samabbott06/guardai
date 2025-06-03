"""
This script demonstrates how to use the GanAPI
to train a GAN model for prompt injection classification.
"""

import warnings
import argparse
from GanAPI import GanAPI

warnings.filterwarnings("ignore", category=UserWarning)

def main():
    # Define and parse command-line args:
    parser = argparse.ArgumentParser(description='Train and use GAN models for prompt injection classification')
    parser.add_argument('--model-name', type=str, default="prompt_injection",
                        help='Name of the model to load or create')
    parser.add_argument('--data-dir', type=str, default="data/train_loaders",
                        help='Directory containing the dataloader files')
    parser.add_argument('--models-dir', type=str, default="models",
                        help='Directory for storing trained models')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save trained models as .pt files (defaults to models-dir)')
    parser.add_argument('--generator-name', type=str, default=None,
                        help='Custom name for the generator model (defaults to model-name_generator)')
    parser.add_argument('--discriminator-name', type=str, default=None,
                        help='Custom name for the discriminator model (defaults to model-name_discriminator)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for training')
    parser.add_argument('--g-lr', type=float, default=0.0002,
                        help='Learning rate for the generator')
    parser.add_argument('--d-lr', type=float, default=0.0002,
                        help='Learning rate for the discriminator')
    parser.add_argument('--smooth-factor', type=float, default=0.1,
                        help='Label smoothing factor (0.0-0.5). Default 0.1 means real=0.9, fake=0.1')
    
    args = parser.parse_args()
    
    # Default values:
    save_dir = args.save_dir if args.save_dir else args.models_dir
    generator_name = args.generator_name if args.generator_name else f"{args.model_name}_generator"
    discriminator_name = args.discriminator_name if args.discriminator_name else f"{args.model_name}_discriminator"
    
    gan = GanAPI(models_dir=args.models_dir,
                 data_dir=args.data_dir,)
    
    print("===== GAN Model for Prompt Injection Classification =====")
    print(f"Using data from: {args.data_dir}")
    print(f"Models directory: {args.models_dir}")
    print(f"Save directory: {save_dir}")
    
    models = gan.list_models()
    print(f"Available models: {models}")
    
    # Train new or load existing models:
    if args.model_name not in models:
        print("\nNo models found. Training a new model...")
        print(f"Generator will be saved as: {generator_name}")
        print(f"Discriminator will be saved as: {discriminator_name}")
        
        generator, discriminator = gan.train(num_epochs=args.epochs,
                                             g_lr=args.g_lr,
                                             d_lr=args.d_lr,
                                             smooth_factor=args.smooth_factor,
                                             noise_dim=64,
                                             hidden_dim=128)
        
        gan.save(args.model_name)
        print(f"Model saved as '{args.model_name}'")
        
        if save_dir:
            import os
            import torch
            
            os.makedirs(save_dir, exist_ok=True)
            gen_path = os.path.join(save_dir, f"{generator_name}.pt")
            disc_path = os.path.join(save_dir, f"{discriminator_name}.pt")
            
            torch.save(generator.state_dict(), gen_path)
            torch.save(discriminator.state_dict(), disc_path)
            
            print(f"Generator saved as: {gen_path}")
            print(f"Discriminator saved as: {disc_path}")
    else:
        print(f"\nLoading model '{args.model_name}'...")
        gan.load(args.model_name)
    
    # Testing generation and classification:
    print("\nGenerating adversarial prompt embeddings...")
    generated_prompts = gan.generate(num_prompts=3)
    print(f"Generated {generated_prompts.shape[0]} prompts with dimension {generated_prompts.shape[1]}")
    
    print("\nClassifying a generated prompt:")
    classification = gan.classify(generated_prompts[0])
    print(f"Classification result: {classification}")
    
    print("\nClassifying a random prompt:")
    random_prompt = torch.randn(8)
    classification = gan.classify(random_prompt)
    print(f"Classification result: {classification}")
    
    print("\nDone")

# Entry Point:
if __name__ == "__main__":
    main()
