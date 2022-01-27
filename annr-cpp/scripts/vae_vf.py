from __future__ import print_function

import torch
import torch.utils.data
from torch import nn, optim, save, load
import numpy as np
import os
from tqdm import tqdm, trange


def load_volume_form(save_folder='checkpoints', latent_dim=2, ver=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    torch.manual_seed(42)

    model_file = os.path.join(save_folder, 'model.pt')
    print('latent dimension: ', latent_dim)

    model = load(model_file)
    model.to(device)
    model.eval()

    print(model)

    def volume_form_v0(x):
        x.requires_grad = True
        x.retain_grad()

        y = model.decode(x).view((len(x), -1))

        jacobian = torch.zeros((len(x), y.shape[-1], latent_dim)).to(device)
        for i in trange(y.shape[-1]):
            model.zero_grad()
            x.grad = None

            y = model.decode(x).view((len(x), -1))
            tmp = torch.zeros(y.shape).to(device)
            tmp[:, i] = torch.ones(len(y))
            y.backward(tmp)
            jacobian[:, i, :] = x.grad
            # print(i)
        metric = jacobian.transpose(-1, -2) @ jacobian
        # print(metric.shape)
        return torch.sqrt(torch.det(metric))

    def volume_form_v1(x):
        jacobian = torch.autograd.functional.jacobian(model.decode, x, create_graph=True).view(len(x), 784, latent_dim)
        metric = jacobian.transpose(-1, -2) @ jacobian
        return torch.sqrt(torch.det(metric))

    def volume_form_numpy_single(x):
        # print(x, x.dtype)
        if ver == 0:
            fn = volume_form_v0
        elif ver == 1:
            fn = volume_form_v1
        return fn(torch.tensor(np.reshape(x, (-1, x.shape[-1])).astype(np.float32), device=device))\
            .detach().cpu().numpy().astype(np.float128).reshape((-1,))

    return volume_form_numpy_single


if __name__ == '__main__':
    import sys
    assert len(sys.argv) > 1
    folder = sys.argv[1]

    import utils
    cfg = utils.load_config(f'{folder}/config.py')

    latent_dim = cfg.dim
    bnd = 1

    from utils import try_load
    n_test = 10000         # TODO hardcoded
    n_grid_points = int(np.ceil(n_test ** (1 / cfg.dim)))
    print(f'Grid with {n_grid_points} points in a dimension')
    grid = try_load(
        f'{folder}/grid_test_data.npy',
        lambda: utils.generate_grid(
            [np.linspace(l, h, n_grid_points) for l, h in \
             zip(cfg.function.domain.lower_limit_vector,
                 cfg.function.domain.upper_limit_vector)],
            cfg.dim))

    volume_form = load_volume_form(ver=0)
    # values = np.array([volume_form(x) for x in tqdm(grid)])
    nblocks = grid.shape[0] // 1000
    print(f'{nblocks=}')
    values = np.concatenate([volume_form(data) for data in np.array_split(grid, nblocks)])
    np.save(f'{folder}/true_grid_test_values.npy', values)
    exit()
