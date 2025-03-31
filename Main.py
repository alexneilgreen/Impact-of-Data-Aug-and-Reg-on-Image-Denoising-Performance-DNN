import os
import argparse
import random
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, STL10

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

def prepare_dataset(dataset_name='MNIST', batch_size=64):
    """
    Prepare dataset with noise for denoising task
    
    Args:
        dataset_name (str): Name of the dataset to use ('MNIST', 'CIFAR10', 'CIFAR100', or 'STL10')
        batch_size (int): Batch size for data loaders
    
    Returns:
        tuple: Train, validation, and test data loaders
    """
    # Define transforms based on the dataset
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Create dataset based on name
    if dataset_name.upper() == 'MNIST':
        # Download MNIST dataset
        train_dataset = MNIST(root='./data', train=True, download=True, transform=basic_transform)
        test_dataset = MNIST(root='./data', train=False, download=True, transform=basic_transform)
        
        # Split train into train and validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    elif dataset_name.upper() == 'CIFAR10':
        # Download CIFAR10 dataset
        train_dataset = CIFAR10(root='./data', train=True, download=True, transform=basic_transform)
        test_dataset = CIFAR10(root='./data', train=False, download=True, transform=basic_transform)
        
        # Split train into train and validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    elif dataset_name.upper() == 'CIFAR100':
        # Download CIFAR100 dataset
        train_dataset = CIFAR100(root='./data', train=True, download=True, transform=basic_transform)
        test_dataset = CIFAR100(root='./data', train=False, download=True, transform=basic_transform)
        
        # Split train into train and validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    elif dataset_name.upper() == 'STL10':
        # Download STL10 dataset
        train_dataset = STL10(root='./data', split='train', download=True, transform=basic_transform)
        test_dataset = STL10(root='./data', split='test', download=True, transform=basic_transform)
        
        # For STL10, we need to handle the split differently as it has unique characteristics
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Custom dataset with noise
    class NoisyDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            if isinstance(self.dataset, torch.utils.data.Subset):
                img, label = self.dataset[idx]
            else:
                img, label = self.dataset[idx]
            
            noisy_img = add_noise(img)
            return noisy_img, img  # Return noisy and clean image pairs
    
    noisy_train = NoisyDataset(train_dataset)
    noisy_val = NoisyDataset(val_dataset)
    noisy_test = NoisyDataset(test_dataset)
    
    train_loader = DataLoader(noisy_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(noisy_val, batch_size=batch_size)
    test_loader = DataLoader(noisy_test, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def get_channels_for_dataset(dataset_name):
    """
    Get the number of input channels based on dataset
    
    Args:
        dataset_name (str): Name of the dataset
    
    Returns:
        int: Number of input channels (1 for grayscale, 3 for RGB)
    """
    if dataset_name.upper() == 'MNIST':
        return 1  # Grayscale
    else:  # CIFAR10, CIFAR100, STL10 are all RGB
        return 3
    
def get_size_for_random_cropping(dataset_name, dataset_sizes):
    """
    Get a size for cropping an image based on its dataset
    
    Args:
        dataset_name (str): Name of the dataset
        dataset_sizes (dict): Maps each dataset name to the length of an image in its set
    
    Returns:
        int: One dimension of the size to be used for the cropped image
    """

    # Randomly select a crop ratio between 0.8 and 0.9
    ratio = random.uniform(0.8, 0.9)

    # Find the dataset
    if dataset_name in dataset_sizes:

        # Determine the new dimensions
        return int(ratio * dataset_sizes[dataset_name])

    # Invalid dataset (will never reach here)
    else:
        return -1

def main(experiment, dataset, epochs, learning_rate):
    """
    Run experiments based on command line arguments
    
    Args:
        experiment (str): Experiment type
        dataset (str): Dataset to use
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
    """
    # Prepare dataset
    train_loader, val_loader, test_loader = prepare_dataset(dataset_name=dataset)

    # Store sizes of each dataset image in case of cropping
    dataset_sizes = {
        'MNIST': 28,
        'CIFAR10': 32,
        'CIFAR100': 32,
        'STL10': 96
    }
    dataset_name_upper = dataset.upper()
    crop_dim = get_size_for_random_cropping(dataset_name_upper, dataset_sizes)
    
    # Augmentation techniques dictionary
    augmentations = {
        'brightness': DataAugmentationTechniques.brightness_adjustment(),
        'color_jitter': DataAugmentationTechniques.color_jitter(),
        'contrast': DataAugmentationTechniques.contrast_modification(),
        'cutout': DataAugmentationTechniques.cutout(),
        'flipping': DataAugmentationTechniques.flipping(),
        'gaussian_noise': DataAugmentationTechniques.gaussian_noise(),
        'random_crop': DataAugmentationTechniques.random_crop(size=crop_dim, resize=dataset_sizes[dataset_name_upper]),
        'rotation': DataAugmentationTechniques.rotation(),
        'scaling': DataAugmentationTechniques.scaling(),
        'shearing': DataAugmentationTechniques.shearing(),
        'custom_augmentation_1': DataAugmentationTechniques.custom_augmentation_1()
    }

    # Initialize Model - adjust based on dataset
    channels = get_channels_for_dataset(dataset)
    # Use in_channels and out_channels to match the Model.py parameters
    model = ImageDenoisingNetwork(in_channels=channels, out_channels=channels)
    
    # Experiment logic
    if experiment == 'base':
        trainer = Trainer(model, train_loader, val_loader, test_loader, learning_rate=learning_rate)
        trainer.train(epochs=epochs, experiment_name=f'base_{dataset.lower()}')
    elif experiment == 'all':
        # Train base model first
        base_trainer = Trainer(model, train_loader, val_loader, test_loader, learning_rate=learning_rate)
        base_trainer.train(epochs=epochs, experiment_name=f'base_{dataset.lower()}')
        
        # Train with individual augmentations
        for aug_name, augmentation in augmentations.items():
            # Reset model for each experiment
            model = ImageDenoisingNetwork(in_channels=channels, out_channels=channels)
            trainer = Trainer(model, train_loader, val_loader, test_loader, augmentation, learning_rate=learning_rate)
            trainer.train(epochs=epochs, experiment_name=f'{aug_name}_{dataset.lower()}')
    elif experiment in augmentations:
        trainer = Trainer(model, train_loader, val_loader, test_loader, augmentations[experiment], learning_rate=learning_rate)
        trainer.train(epochs=epochs, experiment_name=f'{experiment}_{dataset.lower()}')
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
    parser.add_argument('--dataset', type=str, default='MNIST', 
                        choices=['MNIST', 'CIFAR10', 'CIFAR100', 'STL10'],
                        help='Dataset to use (default: MNIST)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer (default: 0.001)')
    
    args = parser.parse_args()
    main(args.experiment, args.dataset, args.epochs, args.learning_rate)