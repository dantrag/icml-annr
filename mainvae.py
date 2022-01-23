from __future__ import print_function

import torch
import torch.utils.data
from torch import nn, optim, save, load
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import pickle

from modelsvae import *



parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help="Set seed for training")
parser.add_argument('--batch-size', type=int, default=64, help="Batch size")
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--val-interval', type=int, default=10)
parser.add_argument('--latent-dim', default=8, type=int, help="Number of classes")

parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--model', default='mlp', type=str, help="Model to use")
parser.add_argument('--dataset', default='mnist', type=str, help="Dataset")
parser.add_argument('--save-interval', default=10, type=int, help="Epoch to save model")
parser.add_argument('--model-name', required=True, type=str, help="Name of model")

args = parser.parse_args()

#GPU
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


batch_size = args.batch_size
epochs = args.epochs
log_interval = args.log_interval
lr = args.lr

latent_dim = args.latent_dim

# Save paths
MODEL_PATH = os.path.join('checkpoints', args.model_name)


figures_dir = os.path.join(MODEL_PATH, 'figures')
tf_log_dir = os.path.join(MODEL_PATH, 'summary')
model_file = os.path.join(MODEL_PATH, 'model.pt')
meta_file = os.path.join(MODEL_PATH, 'metadata.pkl')
log_file = os.path.join(MODEL_PATH, 'log.txt')

make_dir(MODEL_PATH)
make_dir(figures_dir)
make_dir(tf_log_dir)

pickle.dump({'args': args}, open(meta_file, 'wb'))

#Allocate dataset
if args.dataset == 'mnist':
    train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('./files/', train=True, download=True,
                             transform=transforms.ToTensor()),
                batch_size=batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
                datasets.MNIST('./files/', train=False, download=True,
                               transform=transforms.ToTensor()),
                batch_size=batch_size, shuffle=False)


# Sample data
img, _ = next(iter(train_loader))
img_shape = list(img.shape[1:])

# Create Model
if args.model == 'cnn':
    model = VAE_CNN(latent_dim, img_shape[0]).to(device)
elif args.model == 'mlp':
    model = VAE_MLP(1024, latent_dim, img_shape).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)


def train(epoch, data_loader, mode='train'):

    mu_loss = 0
    for batch_idx, (img, _) in enumerate(data_loader):

        if mode == 'train':
            optimizer.zero_grad()
            model.train()
        elif mode == 'val':
            model.eval()

        img = img.to(device)

        recon, mu, logvar = model(img)
        loss = loss_function(recon, img, mu, logvar, beta = 10)
        mu_loss += loss.detach().cpu().numpy()

        if mode == 'train':
            loss.backward()
            optimizer.step()

        if batch_idx % log_interval == 0 and mode == 'train':
            print(f"{mode.upper()} Epoch: {epoch}, Batch: {batch_idx} / {len(train_loader)} Loss: {loss.item():.3}")


    if mode == 'val':
        print(f"{mode.upper()} Epoch: {epoch}, Loss: {(mu_loss / len(data_loader)):.3}")

        # Plot reconstruction
        save_image(recon[:16], f'{figures_dir}/recon_{str(epoch)}.png')


    if (epoch < 10) or (epoch % args.save_interval) == 0:
        save(model, model_file)



if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        train(epoch, train_loader, 'train')
        train(epoch, val_loader, 'val')
