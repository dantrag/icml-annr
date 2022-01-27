import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_path', help='an experiment directory path')
parser.add_argument('--grid_precision', '-g', type=int, default=32)
parser.add_argument('--marginalization_precision', '-m', type=int, default=100000)
parser.add_argument('--iteration', '-i', type=int, default=0)
args = parser.parse_args()

save_path = args.data_path

import utils
cfg = utils.load_config(f'{save_path}/config.py')

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

slices = sum([[(i, j) for j in range(i + 1, cfg.dim)] for i in range(cfg.dim)], [])
print(slices)
methods = ['true', 'annr', 'defer']

g = args.grid_precision
m = args.marginalization_precision

for (i, j) in slices:
    print(f'start {i=} {j=}')
    grids = []
    vmax = 0
    for method in methods:
        folder = f'{save_path}/marg_{method}_{g}_{m}_0'
        if os.path.exists(f'{folder}/grid_{i}_{j}.npy'):
            grid = np.load(f'{folder}/grid_{i}_{j}.npy')
        else:
            print(f'No grid {save_path}/marg_{method}_{g}_{m}_0')
            grid = np.zeros((32, 32))
            for ii in range(32):
                for jj in range(32):
                    if os.path.exists(f'{folder}/values_{i}_{j}_{ii}_{jj}.npy'):
                        grid[ii, jj] = np.mean(np.load(f'{folder}/values_{i}_{j}_{ii}_{jj}.npy'))
        print(f'{method} {np.max(grid)}')
        vmax = max(vmax, np.max(grid))
        grids.append(grid)

    for k, method in enumerate(methods):
        plt.clf()
        plt.figure(figsize=(3, 3))
        plt.imshow(grids[k].T, cmap='Blues', origin='lower', vmax=vmax)
        plt.axis('off')
        plt.gca().set_aspect('equal', 'box')
        plt.tight_layout()
        folder = f'pics/marg_{method}'
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f'{folder}/{i}_{j}.png')
        plt.close()

    print(f'{i=} {j=} {vmax=}')



