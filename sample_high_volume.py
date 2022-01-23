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
import random
from shapely.geometry import Polygon, Point
from scipy.spatial import Voronoi, voronoi_plot_2d



parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str, default='checkpoints', help='Path to saved model')
parser.add_argument('--latent-dim', type=int, default='2', help='Latent dimension')
args_eval = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.manual_seed(42)
torch.cuda.empty_cache()

model_file = os.path.join(args_eval.save_folder, 'model.pt')
latent_dim = args_eval.latent_dim
print('latent dimension: ', latent_dim)

model = load(model_file)
model.eval()

T = 1000
pts = np.load('latent2_lambda_1.0_239/afi_data.npy')[:T]
values = np.load('latent2_lambda_1.0_239/afi_values.npy')[:T]
pts_extended = np.vstack((pts, np.array([[-10., -10.], [ 10., -10.], [-10.,  10.], [ 10.,  10.]]) ))

vor = Voronoi(pts_extended)
vertices = vor.vertices
edge_idxs = vor.ridge_vertices
cells = vor.regions
point_region = vor.point_region

def get_random_point_in_polygon(poly):
    minx, miny, maxx, maxy = poly.bounds
    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if poly.contains(p):
            return torch.FloatTensor(list(p.coords)).to(device)

def voronoi_sample(pts, values):
    idx = np.random.choice(np.array(list(range(len(pts)))), p=(values/values.sum()).astype('float32'))
    cell = cells[point_region[idx]]
    return get_random_point_in_polygon(Polygon(vertices[cell]))

num = 20
fig, axes = plt.subplots(2,num)
for i in range(num):
    z_vor = voronoi_sample(pts, values).unsqueeze(0)
    img_vor = model.decode(z_vor)

    z_gauss = torch.normal(torch.zeros((1,2)), torch.ones((1,2)) ).to(device)
    img_gauss = model.decode(z_gauss)

    ax=axes[1,i]
    ax.imshow( (1-img_vor.view((28,28)).detach().cpu().numpy()) * 255, cmap=plt.cm.binary )
    if i == 0:
        ax.set_title('Voronoi')
    ax.set_axis_off()

    ax=axes[0,i]
    ax.imshow( (1-img_gauss.view((28,28)).detach().cpu().numpy()) * 255, cmap=plt.cm.binary )
    if i == 0:
        ax.set_title('Gaussian')
    ax.set_axis_off()


plt.show()
