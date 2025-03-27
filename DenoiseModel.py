import torch
import torch.nn as nn

class DenoisingUnit(nn.Module):
    def __init__(self, in_channels, expansion_ratio=4):
        super(DenoisingUnit, self).__init__()
        
        # Expansion pointwise convolution (1x1 to increase dimensions)
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion_ratio, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * expansion_ratio),
            nn.ReLU6(inplace=True)
        )
        
        # Depthwise convolution (3x3 convolution applied to each channel)
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels * expansion_ratio, in_channels * expansion_ratio, 
                      kernel_size=3, padding=1, groups=in_channels * expansion_ratio, bias=False),
            nn.BatchNorm2d(in_channels * expansion_ratio),
            nn.ReLU6(inplace=True)
        )
        
        # Pointwise convolution to reduce dimensions back to input channels
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(in_channels * expansion_ratio, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        
    def forward(self, x):
        # Residual connection
        identity = x
        
        # Pass through the denoising unit
        out = self.expand_conv(x)
        out = self.depthwise_conv(out)
        out = self.pointwise_conv(out)
        
        # Add residual connection
        out += identity
        
        return out

class DenoisingNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, num_units=7):
        super(DenoisingNet, self).__init__()
        
        # Initial convolution layer
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU6(inplace=True)
        )
        
        # Denoising units
        self.denoising_units = nn.ModuleList([
            DenoisingUnit(base_channels) for _ in range(num_units)
        ])
        
        # Final convolution layer for reconstruction
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        
        # Shortcut connection between input and output
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Initial convolution
        identity = x
        out = self.initial_conv(x)
        
        # Pass through denoising units
        for unit in self.denoising_units:
            out = unit(out)
        
        # Final convolution
        out = self.final_conv(out)
        
        # Add shortcut connection
        out += self.shortcut(identity)
        
        return out

# Example usage
if __name__ == "__main__":
    # Create the model
    model = DenoisingNet()
    
    # Print model summary
    print(model)
    
    # Example input (batch_size, channels, height, width)
    x = torch.randn(1, 1, 40, 40)
    
    # Forward pass
    output = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)