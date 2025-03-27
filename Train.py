import os
import csv
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np

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
    def random_crop(size=224):
        return transforms.RandomCrop(size)
    
    @staticmethod
    def rotation(degrees=45):
        return transforms.RandomRotation(degrees)
    
    @staticmethod
    def scaling(scale=(0.8, 1.2)):
        return transforms.RandomAffine(0, scale=scale)
    
    @staticmethod
    def shearing(shear=20):
        return transforms.RandomAffine(0, shear=shear)

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
        import pandas as pd
        
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
    
    def train(self, epochs=10, experiment_name='base'):
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
        import json
        with open(os.path.join(results_dir, 'results.json'), 'w') as f:
            json.dump(results, f)

        # Save epoch data to CSV
        self.save_epoch_data(results, results_dir)
        
        return results