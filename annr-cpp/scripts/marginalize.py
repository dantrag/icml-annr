import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_path', help='an experiment directory path')
parser.add_argument('--data_tag', type=str,
                    choices=['annr', 'uniform', 'defer', 'true'], default='annr')
parser.add_argument('--grid_precision', '-g', type=int, default=32)
parser.add_argument('--marginalization_precision', '-m', type=int, default=100000)
parser.add_argument('--iteration', '-i', type=int, default=0)
parser.add_argument('--slice', type=int, default=None)
parser.add_argument('--save', dest='save', action='store_true')
parser.add_argument('--nosave', dest='save', action='store_false')
parser.set_defaults(save=True)
args = parser.parse_args()

save_path = args.data_path
data_tag = args.data_tag

import utils
cfg = utils.load_config(f'{save_path}/config.py')

import annr

import numpy as np
import utils
from utils import timed, try_load, its_own_random
from tqdm import tqdm, trange
import os
import matplotlib.pyplot as plt


def make_compute_values(function, _data):
    def func():
        # _values = Parallel(n_jobs=-1, verbose=5)(delayed(function)(x) for x in _data)
        # return np.array(_values)
        _values = []
        for x in tqdm(_data):
            _values.append(function(x))
        return np.array(_values)
    return func


np.random.seed(cfg.random_seed + 137)
if data_tag in ['annr', 'uniform']:
    if data_tag == 'annr':
        data = np.load(f'{save_path}/{data_tag}_data.npy')
        if data.shape[0] < cfg.n_queries:
            raise Exception(f'Not enough points! {data.shape[0]} out of {cfg.n_queries}')
    elif data_tag == 'uniform':
        with its_own_random(cfg.uniform_data_seed if 'uniform_data_seed' in dir(cfg) else cfg.random_seed + 542):
            data = try_load(f'{save_path}/{data_tag}_data.npy',
                            lambda: utils.uni_sample(cfg.domain, cfg.n_queries))

    values = try_load(f'{save_path}/{data_tag}_values.npy',
                      make_compute_values(cfg.function, data))

    if args.iteration > 0:
        data = data[:args.iteration, :]
        values = values[:args.iteration]

    graph = annr.VoronoiGraph(cfg.strategy)
    graph.initialize(data, annr.Unbounded())

    interpolator = annr.ActiveInterpolator(
        cfg.random_seed + 614, graph, values, cfg.criterion)

    interpolate = interpolator.interpolate

elif data_tag == 'defer' or data_tag == 'defernnr':
    from defer.helpers import Variables, DensityFunctionApproximation, construct
    from defer.variables import Variable
    x = Variable(
        lower=cfg.function.domain.lower_limit_vector,
        upper=cfg.function.domain.upper_limit_vector,
        name="x"
    )
    variables = Variables([x])
    approx: DensityFunctionApproximation = construct(
        fn=cfg.function,
        is_log_fn=False,
        variables=variables,
        num_fn_calls=1,
        callback=lambda i, density:
        print("#Evals: %s. Log Z: %.2f" %
              (density.num_partitions, np.log(density.z))),
        callback_freq_fn_calls=1000,
        is_vectorized_fn=False
    )
    if args.iteration <= 0:
        print(f'Loading {save_path}/defer.pkl')
        approx.load(f'{save_path}/defer.pkl')
    else:
        print(f'Loading {save_path}/extra/{args.iteration}_defer.pkl')
        approx.load(f'{save_path}/extra/{args.iteration}_defer.pkl')
    if data_tag == 'defer':
        interpolate = lambda points: np.array([approx(np.float64(x)) for x in tqdm(points)])
    elif data_tag == 'defernnr':
        data = np.array([.5 * (p.domain.lower_limit_vector + p.domain.upper_limit_vector) for p in approx.all_partitions()]).astype(np.float128)
        values = np.array([p.f for p in approx.all_partitions()]).astype(np.float128)

        graph = annr.VoronoiGraph(cfg.strategy)
        graph.initialize(data, annr.Unbounded())

        interpolator = annr.ActiveInterpolator(
            cfg.random_seed + 614, graph, values, cfg.criterion)
        interpolate = interpolator.interpolate
    else:
        raise Exception()

elif data_tag == 'true':
    interpolate = lambda points: np.array([cfg.function(x) for x in tqdm(points)], dtype=np.float128)

else:
    raise Exception(f'Unknown data tag: {data_tag}')

box = annr.BoundingBox(cfg.function.domain.lower_limit_vector, cfg.function.domain.upper_limit_vector)
lower = box.lower
upper = box.upper

prefix = f'{save_path}/marg_{data_tag}_{args.grid_precision}_{args.marginalization_precision}_{args.iteration}/'
os.makedirs(prefix, exist_ok=True)

def compute_grid(i, j):
    if args.save:
        try_load_this = try_load
    else:
        def try_load_this(s, f):
            print(f'Computing and not saving {s}')
            return f()
    grid = np.zeros((args.grid_precision, args.grid_precision))
    print(f'Parameters: {i} {j}')
    for ii, theta_i in enumerate(np.linspace(lower[i], upper[i], args.grid_precision)):
        for jj, theta_j in enumerate(np.linspace(lower[j], upper[j], args.grid_precision)):
            def gen_data_f():
                with its_own_random(cfg.random_seed + i * 613 + j * 33007 + ii * 14677 + jj * 381111):
                    test_data = utils.uni_sample(box, args.marginalization_precision)
                    test_data[:, i] = theta_i
                    test_data[:, j] = theta_j
                return test_data
            test_data = try_load_this(f'{prefix}/data_{i}_{j}_{ii}_{jj}.npy', gen_data_f)
            test_values = try_load_this(f'{prefix}/values_{i}_{j}_{ii}_{jj}.npy', lambda: interpolate(test_data))
            grid[ii, jj] = np.mean(test_values)
    return grid


if args.slice is not None:
    tmp = args.slice
    i = 0
    j = 1
    while tmp > 0:
        j += 1
        if j >= cfg.dim:
            i += 1
            j = i + 1
        tmp -= 1
    assert(i < cfg.dim)
    try_load(f'{prefix}/grid_{i}_{j}.npy', lambda: compute_grid(i, j))
    exit()

grids = np.zeros((cfg.dim, cfg.dim, args.grid_precision, args.grid_precision))
for i in range(cfg.dim):
    for j in range(i + 1, cfg.dim):
        grids[i, j] = try_load(f'{prefix}/grid_{i}_{j}.npy', lambda: compute_grid(i, j))

vmax = 5

num_plots = cfg.dim * (cfg.dim - 1) // 2
fig, ax = plt.subplots(1, num_plots, figsize=(2 * num_plots, 2))
cur_plot = 1
for i in range(cfg.dim):
    for j in range(i + 1, cfg.dim):
        plt.subplot(1, num_plots, cur_plot)
        plt.axis('off')
        plt.imshow(grids[i, j].T, cmap='Blues', origin='lower', vmax=vmax)
        cur_plot += 1
plt.tight_layout()
plt.savefig(f'{prefix}/marg_{data_tag}_full.png')

picked = [(0, 1), (0, 2), (4, 5)]
num_plots = len(picked)
fig, ax = plt.subplots(1, num_plots, figsize=(2 * num_plots, 2))
cur_plot = 1
for (i, j) in picked:
        plt.subplot(1, num_plots, cur_plot)
        plt.axis('off')
        plt.imshow(grids[i, j].T, cmap='Blues', origin='lower', vmax=vmax)
        cur_plot += 1
plt.tight_layout()
plt.savefig(f'{prefix}/marg_{data_tag}.png')
