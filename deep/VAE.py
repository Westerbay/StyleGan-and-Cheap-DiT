import torch.nn as nn
import torch


class VAE(nn.Module):

    def __init__(self, in_ch, latent_ch):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1), 
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), 
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), 
            nn.ReLU()
        )

        self.mean_pred = nn.Conv2d(256, latent_ch, 1)
        self.logvar_pred = nn.Conv2d(256, latent_ch, 1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, 256, 4, 2, 1),  
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),         
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),          
            nn.ReLU(),
            nn.Conv2d(64, in_ch, 3, 1, 1)
        )

    def encode(self, x):
        h = self.encoder(x)
        mean = self.mean_pred(h)
        logvar = self.logvar_pred(h)
        return mean, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    #Slide 68
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        pred = self.decode(z)
        return pred, mu, logvar
    