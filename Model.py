import torch
import torch.nn as nn

class ImageDenoisingNetwork(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, reg_type=None, dropout_rate=0.2, l1_lambda=1e-5, l2_lambda=1e-4):
        """
        A CNN-based image denoising network with a focus on residual learning
        
        Args:
            in_channels (int): Number of input image channels (default: 1 for grayscale)
            out_channels (int): Number of output image channels (default: 1 for grayscale)
            reg_type (str): Type of regularization ('L1', 'L2', 'DR', 'ES', or None)
            dropout_rate (float): Dropout rate if regularization is 'DR'
            l1_lambda (float): L1 regularization strength
            l2_lambda (float): L2 regularization strength
        """
        super(ImageDenoisingNetwork, self).__init__()
        
        # Regularization parameters passed and stored
        self.reg_type = reg_type
        self.use_dropout = (reg_type == 'DR')

        print(F"Regularization: {reg_type}")
        print(F"Drop Out Rate: {dropout_rate}")
        print(F"L1 Lambda: {l1_lambda}")
        print(F"L2 Lambda: {l2_lambda}")

        # ==================================================
        # Encoder layers
        # ==================================================

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # ==================================================
        # Middle layers
        # ==================================================

        midLayers = [
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        
        # Add dropout after first middle layer if specified
        if self.use_dropout:
            midLayers.append(nn.Dropout2d(p=dropout_rate))
            
        midLayers.extend([
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ])
        
        # Add dropout after second middle layer if specified
        if self.use_dropout:
            midLayers.append(nn.Dropout2d(p=dropout_rate))
            
        self.middle = nn.Sequential(*midLayers)
        
        # ==================================================
        # Decoder layers
        # ==================================================

        decoLayers = [
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        ]
        
        # Add dropout after first decoder layer if specified
        if self.use_dropout:
            decoLayers.append(nn.Dropout2d(p=dropout_rate))
            
        decoLayers.extend([
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ])
        
        # Add dropout after second decoder layer if specified
        if self.use_dropout:
            decoLayers.append(nn.Dropout2d(p=dropout_rate))
            
        decoLayers.append(nn.Conv2d(64, out_channels, kernel_size=3, padding=1))
        
        self.decoder = nn.Sequential(*decoLayers)
        
        # ==================================================

        # For L1/L2 regularization
        self.l1_lambda = l1_lambda  # L1 regularization strength
        self.l2_lambda = l2_lambda  # L2 regularization strength
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input noisy image
        
        Returns:
            torch.Tensor: Denoised image
        """
        # Encoder
        x1 = self.encoder(x)
        
        # Middle
        x2 = self.middle(x1)
        
        # Decoder
        x3 = self.decoder(x2)
        
        return x3
    
    def get_regularization_loss(self):
        """
        Calculate regularization loss based on reg_type
        
        Returns:
            torch.Tensor or None: Regularization loss or None if no regularization
        """
        if self.reg_type == 'L1':
            # L1 regularization (Lasso)
            l1_reg = torch.tensor(0., device=next(self.parameters()).device)
            for param in self.parameters():
                l1_reg += torch.sum(torch.abs(param))
            return self.l1_lambda * l1_reg
        
        elif self.reg_type == 'L2':
            # L2 regularization (Ridge)
            l2_reg = torch.tensor(0., device=next(self.parameters()).device)
            for param in self.parameters():
                l2_reg += torch.sum(param ** 2)
            return self.l2_lambda * l2_reg
        
        else:
            # No regularization
            return None