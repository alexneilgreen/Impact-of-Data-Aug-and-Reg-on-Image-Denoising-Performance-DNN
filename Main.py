import os
import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

from Model import ImageDenoisingNetwork
from Train import Trainer, DataAugmentationTechniques

def add_noise(x, noise_factor=0.5):
    """
    Add Gaussian noise to input tensor
    
    Args:
        x (torch.Tensor): Input tensor
        noise_factor (float): Noise intensity
    
    Returns:
        torch.Tensor: Noisy tensor
    """
    noisy = x + noise_factor * torch.randn_like(x)
    noisy = torch.clamp(noisy, 0., 1.)
    return noisy

def prepare_dataset(batch_size=64):
    """
    Prepare MNIST dataset with noise for denoising task
    
    Args:
        batch_size (int): Batch size for data loaders
    
    Returns:
        tuple: Train, validation, and test data loaders
    """
    # Download MNIST dataset
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    
    # Split train into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Custom dataset with noise
    class NoisyDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            noisy_img = add_noise(img)
            return noisy_img, img
    
    noisy_train = NoisyDataset(train_dataset)
    noisy_val = NoisyDataset(val_dataset)
    noisy_test = NoisyDataset(test_dataset)
    
    train_loader = DataLoader(noisy_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(noisy_val, batch_size=batch_size)
    test_loader = DataLoader(noisy_test, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def main(experiment, epochs, learning_rate):
    """
    Run experiments based on command line arguments
    
    Args:
        experiment (str): Experiment type
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
    """
    # Prepare dataset
    train_loader, val_loader, test_loader = prepare_dataset()
    
    # Augmentation techniques dictionary
    augmentations = {
        'brightness': DataAugmentationTechniques.brightness_adjustment(),
        'color_jitter': DataAugmentationTechniques.color_jitter(),
        'contrast': DataAugmentationTechniques.contrast_modification(),
        'cutout': DataAugmentationTechniques.cutout(),
        'flipping': DataAugmentationTechniques.flipping(),
        'gaussian_noise': DataAugmentationTechniques.gaussian_noise(),
        'random_crop': DataAugmentationTechniques.random_crop(),
        'rotation': DataAugmentationTechniques.rotation(),
        'scaling': DataAugmentationTechniques.scaling(),
        'shearing': DataAugmentationTechniques.shearing(),
        'custom_augmentation_1': DataAugmentationTechniques.custom_augmentation_1()
    }

    # Initialize Model
    model = ImageDenoisingNetwork()
    
    # Experiment logic
    if experiment == 'base':
        trainer = Trainer(model, train_loader, val_loader, test_loader, learning_rate=learning_rate)
        trainer.train(epochs=epochs, experiment_name='base')
    elif experiment == 'all':
        # Train base model first
        base_trainer = Trainer(model, train_loader, val_loader, test_loader, learning_rate=learning_rate)
        base_trainer.train(epochs=epochs, experiment_name='base')
        
        # Train with individual augmentations
        for aug_name, augmentation in augmentations.items():
            # Reset model for each experiment
            model = ImageDenoisingNetwork()
            trainer = Trainer(model, train_loader, val_loader, test_loader, augmentation, learning_rate=learning_rate)
            trainer.train(epochs=epochs, experiment_name=aug_name)
    elif experiment in augmentations:
        trainer = Trainer(model, train_loader, val_loader, test_loader, augmentations[experiment], learning_rate=learning_rate)
        trainer.train(epochs=epochs, experiment_name=experiment)
    else:
        raise ValueError(f"Invalid experiment: {experiment}")

if __name__ == "__main__":
    # Augmentation techniques dictionary (pre-defined for argparse choices)
    augmentations = {
        'brightness', 'color_jitter', 'contrast', 'cutout', 
        'flipping', 'gaussian_noise', 'random_crop', 
        'rotation', 'scaling', 'shearing', 'custom_augmentation_1'
    }

    parser = argparse.ArgumentParser(description='Image Denoising Experiments')
    parser.add_argument('--experiment', type=str, default='base',
                        choices=['base', 'all'] + list(augmentations),
                        help='Experiment type or specific augmentation')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer (default: 0.001)')
    
    args = parser.parse_args()
    main(args.experiment, args.epochs, args.learning_rate)