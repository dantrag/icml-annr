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
from matplotlib.pyplot import imshow


parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str, default='checkpoints', help='Path to saved model')
parser.add_argument('--latent-dim', type=int, default='2', help='Latent dimension')
args_eval = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.cuda.empty_cache()

model_file = os.path.join(args_eval.save_folder, 'model.pt')
latent_dim = args_eval.latent_dim
print('latent dimension: ', latent_dim)

model = load(model_file)
model.eval()

def show_image(x, row):
    img = model.decode(x)
    for i in range(1,11):
        fig.add_subplot(row, 10, i)
        plt.axis('off')
        plt.imshow( (1-img[i].view((28,28)).detach().cpu().numpy()) * 255, cmap=plt.cm.binary )
    #plt.show()
    #plt.savefig('mnist_early.png', dpi=500)


def volume_form(x):
    x.requires_grad = True
    x.retain_grad()

    y = model.decode(x).view((len(x), -1))
    jacobian = torch.zeros((len(x), y.shape[-1], latent_dim)).to(device)
    for i in range(y.shape[-1]):
        model.zero_grad()
        x.grad = None

        y = model.decode(x).view((len(x), -1))
        tmp = torch.zeros(y.shape).to(device)
        tmp[:, i] = torch.ones(len(y))
        y.backward(tmp)
        jacobian[:,i,:] = x.grad
        print(i)
    metric = jacobian.transpose(-1,-2) @ jacobian
    #print(metric.shape)
    return torch.sqrt(torch.det(metric))

lim = 5
res = 50
x = torch.linspace(-lim,lim,res)
y = torch.linspace(lim,-lim,res)  #goes the other way around because of matrix conventions
xx, yy = torch.meshgrid(x,y)
grid = torch.cat((xx.reshape((-1,1)), yy.reshape((-1,1))),dim=-1).to(device)
if latent_dim > 2:
    grid = torch.cat((grid, torch.zeros((len(grid), latent_dim-2)).to(device)), dim=-1)


values = volume_form(grid)
print(values.mean(), values.max())

pts1 = np.load('latent2_lambda_1.0_239/afi_data.npy')[480:500]
pts2 = np.load('latent2_lambda_1.0_239/afi_data.npy')[980:1000]

fig, ax = plt.subplots(1,1)
#ax.scatter(pts[:,0], pts[:,1], c='Black', alpha=.9, zorder=2, s=20.)

imshow(values.reshape((res,res)).T.detach().cpu().numpy() , cmap='Blues', extent=(-1,1,-1,1), alpha=.9)

plt.xlim(-1,1)
plt.ylim(-1,1)
ax.set_aspect('equal', adjustable='box')
plt.axis('off')
plt.savefig(f'latent_picture.png', dpi=500, bbox_inches='tight', pad_inches=0.05)

plt.show()

# fig = plt.figure()
# show_image(torch.FloatTensor(pts1.astype('float32')).to(device), 2)
# show_image(torch.FloatTensor(pts2.astype('float32')).to(device), 1)
# plt.show()
