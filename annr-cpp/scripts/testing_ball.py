import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_path', help='an experiment directory path')
parser.add_argument('--data_tag', type=str,
                    choices=['annr', 'uniform', 'defer', 'none'], default='annr')
parser.add_argument('--test_tag', type=str,
                    choices=['grid', 'uniform'], default='uniform')
parser.add_argument('--iteration', '-i', type=int, default=0)
parser.add_argument('--radius', '-r', type=float, default=None)
parser.add_argument('--mae_out', type=str, default=None)
parser.add_argument('--exec', type=str, default=None)
parser.add_argument('--n_test_multiplier', type=float, default=10)
args = parser.parse_args()


assert args.radius is None or args.test_tag != 'grid'

# if len(sys.argv) <= 1:
#     raise Exception('Please provide an experiment directory path')
save_path = args.data_path
data_tag = args.data_tag
test_tag = args.test_tag

import utils
cfg = utils.load_config(f'{save_path}/config.py')

import annr

import numpy as np
import utils
from utils import timed, try_load, its_own_random
from tqdm import tqdm, trange
import os
#from joblib import Parallel, delayed, cpu_count
from multiprocessing import Pool, cpu_count


if args.exec is not None:
    exec(args.exec)

n_test = int(cfg.n_queries * args.n_test_multiplier)


def mae(a, b):
    ids = np.isfinite(a) & np.isfinite(b)
    return np.mean(np.abs(a[ids] - b[ids]))


def make_compute_values(function, _data):
    def func():
        # _values = Parallel(n_jobs=-1, verbose=5)(delayed(function)(x) for x in _data)
        # return np.array(_values)
        _values = []
        for x in tqdm(_data):
            _values.append(function(x))
        return np.array(_values)
    return func


print(f'Function: {cfg.function.name}')
print(f'Bounds: {cfg.domain}')
print(f'Space dimensionality: {cfg.dim}')
if args.iteration > 0:
    print(f'Subsample size: {args.iteration}')

if args.iteration > 0:
    prefix = f'extra/{args.iteration}_'
    if args.radius is None:
        os.makedirs(f'{save_path}/extra', exist_ok=True)
    else:
        os.makedirs(f'{save_path}/sphere_test/extra', exist_ok=True)

else:
    prefix = ''

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
        cfg.random_seed + 614, graph, values.astype(np.float128), cfg.criterion)

    interpolate = interpolator.interpolate

elif data_tag == 'defer':
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
    print(f'Loading {save_path}/defer.pkl')
    approx.load(f'{save_path}/defer.pkl')
    interpolate = lambda points: np.array([approx(np.float64(x)) for x in tqdm(points)])

elif data_tag == 'none':
    interpolate = None
else:
    raise Exception(f'Unknown data tag: {data_tag}')

assert test_tag == 'uniform'
with its_own_random(cfg.test_seed if 'test_seed' in dir(cfg) else cfg.random_seed + 724):
    if args.radius is None:
        test_data = try_load(
            f'{save_path}/{test_tag}_test_data.npy',
            lambda: utils.uni_sample(cfg.domain, n_test)).astype(np.float128)
        assert test_data.shape[0] == n_test, \
            f'Number of test points is different from desired, f{test_data.shape[0]} vs {n_test}'
    else:
        os.makedirs(f'{save_path}/sphere_test', exist_ok=True)
        test_data = try_load(
            f'{save_path}/sphere_test/test_data_{args.radius}.npy',
            lambda: utils.gen_sphere(n_test, cfg.dim, args.radius, dtype=np.float128))

if args.radius is None:
    true_test_values = try_load(
        f'{save_path}/true_{test_tag}_test_values.npy',
        make_compute_values(cfg.function, test_data))

    if interpolate is None:
        exit()

    approximate_test_values = try_load(
        f'{save_path}/{prefix}{data_tag}_{test_tag}_test_values.npy',
        lambda: interpolate(test_data))
else:
    true_test_values = try_load(
        f'{save_path}/sphere_test/true_test_values_{args.radius}.npy',
        make_compute_values(cfg.function, test_data))

    if interpolate is None:
        exit()

    approximate_test_values = try_load(
        f'{save_path}/sphere_test/{prefix}{data_tag}_test_values_{args.radius}.npy',
        lambda: interpolate(test_data))


mae_val = mae(true_test_values, approximate_test_values)
print(f'{args.iteration=}\ndata: {data_tag}\ntest: {test_tag}\nMAE: {mae_val}')

if args.mae_out is not None:
    with open(args.mae_out, 'a') as f:
        f.write(f'"{save_path}" {args.iteration} {args.radius} {data_tag} {test_tag} {mae_val}\n')
