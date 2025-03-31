import os
import csv
import json
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class DataAugmentationTechniques:
    """
    Collection of data augmentation techniques for image denoising
    """
    @staticmethod
    def brightness_adjustment(brightness_factor=1.5):
        return transforms.ColorJitter(brightness=brightness_factor)
    
    @staticmethod
    def color_jitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        return transforms.ColorJitter(brightness, contrast, saturation, hue)
    
    @staticmethod
    def contrast_modification(contrast_factor=1.5):
        return transforms.ColorJitter(contrast=contrast_factor)
    
    @staticmethod
    def cutout(p=0.5, size=20):
        return transforms.RandomErasing(p=p, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
    
    @staticmethod
    def flipping():
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0)
        ])
    
    @staticmethod
    def gaussian_noise(mean=0., std=0.1):
        def noise_transform(tensor):
            return tensor + torch.randn_like(tensor) * std + mean
        return noise_transform
    
    @staticmethod
    def random_crop(size=224, resize=224):
        return transforms.Compose([
            transforms.RandomCrop(size),
            transforms.Resize(resize)
        ])
    
    @staticmethod
    def rotation(degrees=45):
        return transforms.RandomRotation(degrees)
    
    @staticmethod
    def scaling(scale=(0.8, 1.2)):
        return transforms.RandomAffine(0, scale=scale)
    
    @staticmethod
    def shearing(shear=20):
        return transforms.RandomAffine(0, shear=shear)
    
    @staticmethod
    def custom_augmentation_1():
        """
        Create a custom augmentation transform
        
        Returns:
            transforms.Compose: A composition of augmentation transforms
        """
        def rescale_transform(x):
            # Generate random scaling factor between 0.9 and 1.1
            scale = 0.9 + torch.rand(1).item() * 0.2
            return x * scale
        
        return transforms.Compose([
            # Rescale image to range (0.9, 1.1)
            transforms.Lambda(rescale_transform),
            
            # Rotate image with angle range (0, 20)
            transforms.RandomRotation(degrees=(0, 20)),
            
            # Flip vertically with probability 0.5
            transforms.RandomVerticalFlip(p=0.5),
            
            # Convert to grayscale with probability 0.5
            transforms.RandomGrayscale(p=0.5)
        ])

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, augmentation=None, learning_rate=0.001):
        """
        Initialize trainer with model, data loaders, and optional augmentation
        
        Args:
            model (nn.Module): Neural network model
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            test_loader (DataLoader): Test data loader
            augmentation (callable, optional): Data augmentation transform
            learning_rate (float, optional): Learning rate for optimizer
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.augmentation = augmentation
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def save_epoch_data(self, results, results_dir):
        """
        Save epoch-level data to CSV in the results directory
        
        Args:
            results (dict): Experiment results
            results_dir (str): Directory to save results
        """        
        # Prepare epoch data
        epoch_data = pd.DataFrame({
            'Epoch': range(1, len(self.train_losses) + 1),
            'Train Loss': self.train_losses,
            'Validation Loss': self.val_losses,
            'Train Accuracy': self.train_accuracies,
            'Validation Accuracy': self.val_accuracies
        })
        
        # Save to CSV
        csv_path = os.path.join(results_dir, 'epoch_data.csv')
        epoch_data.to_csv(csv_path, index=False)
    
    def visualize_denoising_results(self, num_samples=5, experiment_name='base'):
        """
        Generate side-by-side comparisons of noisy and denoised images
        
        Args:
            num_samples (int): Number of image samples to visualize
            experiment_name (str): Name of the experiment
        """
        # Create save directory following the specified format
        save_dir = os.path.join('Results', experiment_name, 'visual_comparisons')
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Disable gradient computation
        with torch.no_grad():
            for batch_idx, (noisy_images, clean_images) in enumerate(self.test_loader):
                # Only process first batch
                if batch_idx >= 1:
                    break
                
                # Process num_samples images from the batch
                for i in range(min(num_samples, noisy_images.size(0))):
                    # Get individual images
                    noisy_img = noisy_images[i]
                    clean_img = clean_images[i]
                    
                    # Move images to the same device as the model
                    noisy_img = noisy_img.to(self.device)
                    clean_img = clean_img.to(self.device)
                    
                    # Denoise the image
                    denoised_img = self.model(noisy_img.unsqueeze(0)).squeeze(0)
                    
                    # Move images back to CPU for visualization
                    noisy_np = noisy_img.cpu().numpy()
                    clean_np = clean_img.cpu().numpy()
                    denoised_np = denoised_img.cpu().numpy()
                    
                    # Create subplot
                    plt.figure(figsize=(15, 6))
                    
                    # Handle different channel structures
                    if clean_np.shape[0] == 1:  # Grayscale
                        # Remove channel dimension for plotting
                        clean_np = clean_np.squeeze(0)
                        noisy_np = noisy_np.squeeze(0)
                        denoised_np = denoised_np.squeeze(0)
                        
                        plt.subplot(1, 3, 1)
                        plt.title('Clean Image')
                        plt.imshow(clean_np, cmap='gray')
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 2)
                        plt.title('Noisy Image')
                        plt.imshow(noisy_np, cmap='gray')
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 3)
                        plt.title('Denoised Image')
                        plt.imshow(denoised_np, cmap='gray')
                        plt.axis('off')
                    else:  # RGB
                        # Transpose from (C,H,W) to (H,W,C) format for matplotlib
                        clean_np = np.transpose(clean_np, (1, 2, 0))
                        noisy_np = np.transpose(noisy_np, (1, 2, 0))
                        denoised_np = np.transpose(denoised_np, (1, 2, 0))
                        
                        plt.subplot(1, 3, 1)
                        plt.title('Clean Image')
                        plt.imshow(clean_np)
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 2)
                        plt.title('Noisy Image')
                        plt.imshow(noisy_np)
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 3)
                        plt.title('Denoised Image')
                        plt.imshow(denoised_np)
                        plt.axis('off')
                    
                    # Save the figure
                    plt.tight_layout()
                    save_path = os.path.join(save_dir, f'comparison_{batch_idx}_{i}.png')
                    plt.savefig(save_path)
                    plt.close()
    
    def FID_images(self, experiment_name='base', num_samples=5):
        """
        Save original and generated images for FID calculation
        
        Args:
            experiment_name (str): Name of the experiment for result saving
            num_samples (int): Number of image samples to save
        """
        # Create save directory following the specified format
        save_dir = os.path.join('Results', experiment_name, 'images')
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Disable gradient computation
        with torch.no_grad():
            for batch_idx, (noisy_images, clean_images) in enumerate(self.test_loader):
                # Only process first batch
                if batch_idx >= 1:
                    break
                
                # Process num_samples images from the batch
                for i in range(min(num_samples, noisy_images.size(0))):
                    # Get individual images
                    noisy_img = noisy_images[i]
                    clean_img = clean_images[i]
                    
                    # Move images to the same device as the model
                    noisy_img = noisy_img.to(self.device)
                    clean_img = clean_img.to(self.device)
                    
                    # Denoise the image
                    denoised_img = self.model(noisy_img.unsqueeze(0)).squeeze(0)
                    
                    # Move images back to CPU for saving
                    clean_np = clean_img.cpu().numpy()
                    denoised_np = denoised_img.cpu().numpy()
                    
                    # Handle different channel structures
                    if clean_np.shape[0] == 1:  # Grayscale (1 channel)
                        clean_np = clean_np.squeeze(0)  # Remove channel dimension for grayscale
                        denoised_np = denoised_np.squeeze(0)
                        # Normalize images to 0-255 range for saving
                        clean_np = (clean_np * 255).astype(np.uint8)
                        denoised_np = (denoised_np * 255).astype(np.uint8)
                        
                        # Save using PIL
                        Image.fromarray(clean_np).save(os.path.join(save_dir, f'{i+1}_original.png'))
                        Image.fromarray(denoised_np).save(os.path.join(save_dir, f'{i+1}_generated.png'))
                    else:  # RGB (3 channels)
                        # Transpose from (C,H,W) to (H,W,C) format for PIL
                        clean_np = np.transpose(clean_np, (1, 2, 0))
                        denoised_np = np.transpose(denoised_np, (1, 2, 0))
                        
                        # Normalize images to 0-255 range for saving
                        clean_np = (clean_np * 255).astype(np.uint8)
                        denoised_np = (denoised_np * 255).astype(np.uint8)
                        
                        # Save original and generated images
                        original_path = os.path.join(save_dir, f'{i+1}_original.png')
                        generated_path = os.path.join(save_dir, f'{i+1}_generated.png')
                        
                        # Use PIL to save images
                        Image.fromarray(clean_np).save(original_path)
                        Image.fromarray(denoised_np).save(generated_path)

    def train(self, epochs=50, experiment_name='base'):
        """
        Train the model and track performance metrics
        
        Args:
            epochs (int): Number of training epochs
            experiment_name (str): Name of the experiment for result saving
        
        Returns:
            dict: Training metrics
        """
        # Create results directory
        results_dir = os.path.join('Results', experiment_name)
        os.makedirs(results_dir, exist_ok=True)
        
        # Lists to track training progress - RESET these BEFORE training
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Open log file
        log_file_path = os.path.join(results_dir, 'trainingLog.txt')
        log_file = open(log_file_path, 'w')
        
        def log_and_print(message):
            """
            Log message to both console and file
            
            Args:
                message (str): Message to log
            """
            print(message)
            log_file.write(message + '\n')
            log_file.flush()  # Ensure message is written immediately
        
        start_time = time.time()
        
        log_and_print(f"Starting training for experiment: {experiment_name}")
        log_and_print(f"Total epochs: {epochs}")
        log_and_print(f"Device: {self.device}")
        
        for epoch in range(1, epochs + 1):
            # Training phase
            self.model.train()
            epoch_train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                if self.augmentation:
                    data = self.augmentation(data)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
                train_correct += ((output - target).abs() < 0.1).float().mean().item()
                train_total += 1
            
            train_loss = epoch_train_loss / len(self.train_loader)
            train_accuracy = train_correct / train_total
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            
            # Validation phase
            self.model.eval()
            epoch_val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    epoch_val_loss += loss.item()
                    val_correct += ((output - target).abs() < 0.1).float().mean().item()
                    val_total += 1
            
            val_loss = epoch_val_loss / len(self.val_loader)
            val_accuracy = val_correct / val_total
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Log epoch results
            log_and_print(f"Epoch {epoch}/{epochs}")
            log_and_print(f"  Train Loss:\t\t{train_loss:.4f}\tTrain Accuracy:\t\t{train_accuracy:.4f}")
            log_and_print(f"  Validation Loss:\t{val_loss:.4f}\tValidation Accuracy:\t{val_accuracy:.4f}")
            log_and_print("-" * 50)
        
        # Test phase
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                test_loss += loss.item()
                test_correct += ((output - target).abs() < 0.1).float().mean().item()
                test_total += 1
        
        test_loss /= len(self.test_loader)
        test_accuracy = test_correct / test_total
        
        total_time = time.time() - start_time

        # Generate FID Images
        self.FID_images(experiment_name=experiment_name)
        # Generate Visual Example
        self.visualize_denoising_results(experiment_name=experiment_name)
        
        # Log final results
        log_and_print("\nFinal Test Results:")
        log_and_print(f"Test Loss: {test_loss:.4f}")
        log_and_print(f"Test Accuracy: {test_accuracy:.4f}")
        log_and_print(f"Total Computation Time: {total_time:.2f} seconds")
        
        # Close log file
        log_file.close()
        
        # Save results
        results = {
            'train_losses': train_loss,
            'val_losses': val_loss,
            'train_accuracies': train_accuracy,
            'val_accuracies': val_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'computation_time': total_time
        }
        
        # Save results to JSON
        with open(os.path.join(results_dir, 'results.json'), 'w') as f:
            json.dump(results, f)

        # Save epoch data to CSV
        self.save_epoch_data(results, results_dir)
        
        return results