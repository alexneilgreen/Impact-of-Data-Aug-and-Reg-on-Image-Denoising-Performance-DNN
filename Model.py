import torch
import torch.nn as nn

class ImageDenoisingNetwork(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        """
        A CNN-based image denoising network with a focus on residual learning
        
        Args:
            in_channels (int): Number of input image channels (default: 1 for grayscale)
            out_channels (int): Number of output image channels (default: 1 for grayscale)
        """
        super(ImageDenoisingNetwork, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Middle layers
        self.middle = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )
    
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