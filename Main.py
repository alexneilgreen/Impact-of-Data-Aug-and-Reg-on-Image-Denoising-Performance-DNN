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

def prepare_dataset(dataset_name='MNIST', batch_size=64, noise_factor=0.5):
    """
    Prepare dataset with noise for denoising task
    
    Args:
        dataset_name (str): Name of the dataset to use ('MNIST', 'CIFAR10', 'CIFAR100', or 'STL10')
        batch_size (int): Batch size for data loaders
        noise_factor (float): Noise factor to be applied to tensors
    
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
            
            noisy_img = add_noise(img, noise_factor)
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

def main(experiment, dataset, epochs, learning_rate, results_dir_base, noise, reg_val=None, drop_rate=0.2, l1_lambda=1e-5, l2_lambda=1e-4):
    """
    Run experiments based on command line arguments
    
    Args:
        experiment (str): Experiment type
        dataset (str): Dataset to use
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        results_dir_base (str): Base name for folder to store results
        noise (float): Noise level to be applied to tensors
        reg_val (str): Type of regularization to apply ('L1', 'L2', 'Dr', 'ES', 'all', or None)
        drop_rate (float): Dropout rate for dropout regularization
        l1_lambda (float): L1 regularization strength
        l2_lambda (float): L2 regularization strength
    """
    # Prepare dataset
    train_loader, val_loader, test_loader = prepare_dataset(dataset_name=dataset, noise_factor=noise)

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

    # Get number of channels based on dataset
    channels = get_channels_for_dataset(dataset)
    
    # Experiment logic
    if experiment == 'base':
        # Check if we should run all regularization types
        if reg_val == 'all':
            # Define regularization options to loop through
            reg_options = ['L1', 'L2', 'DR', 'ES', None]
            
            for current_reg in reg_options:
                # Create a new model instance for each regularization type
                model = ImageDenoisingNetwork(in_channels=channels, out_channels=channels, 
                                             reg_type=current_reg, dropout_rate=drop_rate,
                                             l1_lambda=l1_lambda, l2_lambda=l2_lambda)
                
                # Create directory name with the current regularization
                reg_suffix = f"_{current_reg}" if current_reg else ""
                current_results_dir = f'{results_dir_base}_{noise}{reg_suffix}'
                
                trainer = Trainer(model, train_loader, val_loader, test_loader, 
                                 learning_rate=learning_rate, 
                                 results_dir_base=current_results_dir)
                
                trainer.train(epochs=epochs, experiment_name=f'base_{dataset.lower()}_reg_{current_reg if current_reg else "none"}')
        else:
            # Initialize Model with specific regularization
            model = ImageDenoisingNetwork(in_channels=channels, out_channels=channels, 
                                         reg_type=reg_val, dropout_rate=drop_rate,
                                         l1_lambda=l1_lambda, l2_lambda=l2_lambda)
            
            # Create directory name
            reg_suffix = f"_{reg_val}" if reg_val else ""
            results_dir_base_with_params = f'{results_dir_base}_{noise}{reg_suffix}'
            
            trainer = Trainer(model, train_loader, val_loader, test_loader, 
                             learning_rate=learning_rate, 
                             results_dir_base=results_dir_base_with_params)
            
            trainer.train(epochs=epochs, experiment_name=f'base_{dataset.lower()}')
    
    elif experiment == 'all':
        # For 'all' experiment, ignore regVal and proceed with just the experiment
        # Create directory with just noise level
        results_dir_base_with_params = f'{results_dir_base}_{noise}'
        
        # Train base model first without any regularization
        model = ImageDenoisingNetwork(in_channels=channels, out_channels=channels)
        base_trainer = Trainer(model, train_loader, val_loader, test_loader, 
                              learning_rate=learning_rate, 
                              results_dir_base=results_dir_base_with_params)
        base_trainer.train(epochs=epochs, experiment_name=f'base_{dataset.lower()}')
        
        # Train with individual augmentations
        for aug_name, augmentation in augmentations.items():
            # Reset model for each experiment
            model = ImageDenoisingNetwork(in_channels=channels, out_channels=channels, drop_rate=drop_rate)
            trainer = Trainer(model, train_loader, val_loader, test_loader, 
                             augmentation, learning_rate=learning_rate, 
                             results_dir_base=results_dir_base_with_params)
            trainer.train(epochs=epochs, experiment_name=f'{aug_name}_{dataset.lower()}')
    
    elif experiment in augmentations:
        # For specific augmentation experiments, ignore regVal and proceed with just the experiment
        # Create directory with just noise level
        results_dir_base_with_params = f'{results_dir_base}_{noise}'
        
        model = ImageDenoisingNetwork(in_channels=channels, out_channels=channels)
        trainer = Trainer(model, train_loader, val_loader, test_loader, 
                         augmentations[experiment], learning_rate=learning_rate, 
                         results_dir_base=results_dir_base_with_params)
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
    
    # Regularization techniques
    regularization_types = ['L1', 'L2', 'DR', 'ES', None]

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
    parser.add_argument('--output_dir', type=str, default='Results',
                        help='Name of output directory base folder')
    parser.add_argument('--noise', type=float, default=0.5,
                        help='Noise level in range [0.0, 1.0]')
    parser.add_argument('--regVal', type=str, default=None,
                        choices=['L1', 'L2', 'DR', 'ES', 'None', 'all'],
                        help='Regularization type: L1 (L1 Regularization), L2 (L2 Regularization), ' +
                             'DR (Dropout), ES (Early Stopping), all (try all types), or None (default)')
    parser.add_argument('--dropRate', type=float, default=0.2, choices=[0.2, 0.3, 0.4],
                        help='Dropout rate for dropout regularization (default: 0.2)')
    parser.add_argument('--L1', type=float, default=1e-5, choices=[1e-5, 1e-4, 1e-3],
                        help='L1 regularization strength (default: 1e-5)')
    parser.add_argument('--L2', type=float, default=1e-4, choices=[1e-4, 1e-3, 1e-2],
                        help='L2 regularization strength (default: 1e-4)')
    
    args = parser.parse_args()
    
    # Convert 'None' string to None type
    reg_val = None if args.regVal == 'None' else args.regVal
    
    main(args.experiment, args.dataset, args.epochs, args.learning_rate, 
         args.output_dir, args.noise, reg_val, args.dropRate, args.L1, args.L2)