import torch
import torch.nn as nn

class ImageDenoisingVAE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_filters=32, latent_dim=64, input_size_1_dim = 150):
        """
        A CNN-based image denoising Variational Autoencoder

        Args:
            in_channels (int): Number of input image channels (default: 1 for grayscale)
            out_channels (int): Number of output image channels (default: 1 for grayscale)
            num_filters (int): Number of filters in convolutional layers
            latent_dim (int): Dimension of the latent space
        """
        super(ImageDenoisingVAE, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
            nn.Flatten()
        )

        # Compute the flattened size dynamically
        flattened_1_dim = input_size_1_dim // 8
        flattened_size = flattened_1_dim * num_filters

        # Latent space layers
        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size, latent_dim)

        # Decoder preparation layers
        self.fc_decoder = nn.Linear(latent_dim, flattened_size)
        self.unflatten = nn.Unflatten(1, (num_filters, flattened_1_dim, flattened_1_dim))
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_filters, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_filters, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_filters, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_filters, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + sigma * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        """Encodes input into a latent representation."""
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def decode(self, z):
        """Decodes latent representation back to image."""
        x = self.fc_decoder(z)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x

    def forward(self, x):
        """Full forward pass: Encode → Reparameterize → Decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar