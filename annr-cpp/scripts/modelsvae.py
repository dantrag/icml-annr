import torch
from torch import nn, load
from torch.nn import functional as F
from torchvision.utils import save_image
from torch import linalg as LA
import numpy as np
from functools import reduce


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)



class VAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    def encode(self, x):
        encoded =  self.encoder(x)
        return encoded[:, :self.latent_dim], encoded[:, self.latent_dim : ]


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):                           #Monte Carlo (with one sample)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar





class VAE_MLP(VAE):
    def __init__(self, int_neur, latent_dim, img_shape):
        pixels = reduce((lambda x, y: x * y), img_shape)
        encoder = nn.Sequential(
            View([-1, pixels]),
            nn.Linear(pixels, int_neur),
            nn.ReLU(True),
            nn.Linear(int_neur, int_neur),
            nn.ReLU(True),
            nn.Linear(int_neur, 2*latent_dim),
        )
        decoder = nn.Sequential(
            nn.Linear(latent_dim, int_neur),
            nn.ReLU(True),
            nn.Linear(int_neur, int_neur),
            nn.ReLU(True),
            nn.Linear(int_neur, pixels),
            View([-1] + img_shape),
            nn.Sigmoid()
            )
        super(VAE_MLP, self).__init__(encoder, decoder, latent_dim)




class VAE_CNN(VAE):                           #64x64
    def __init__(self, latent_dim, nc):

        encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, latent_dim*2),             # B, latent_dim*2
        )
        decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
            nn.Sigmoid()
            )
        super(VAE_CNN, self).__init__(encoder, decoder, latent_dim)



def recon_loss(recon_x, x):
    return F.binary_cross_entropy(recon_x, x, reduction = 'sum')


def loss_function(recon_x, x, mu, logvar, beta = 1):
    KL_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss(recon_x, x) + beta * KL_loss
