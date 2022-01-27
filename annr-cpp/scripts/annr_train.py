import argparse
parser = argparse.ArgumentParser()
parser.add_argument('config_path', help='a configuration file path')
args = parser.parse_args()

import utils
cfg = utils.load_config(args.config_path)

import pickle
import shutil
import filecmp
import annr

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import math
from time import time
import os
from utils import timed, try_load, load_random, uni_sample
import utils

print(f'Function: {cfg.function.name}')
print(f'Bounds: {cfg.domain}')
print(f'Space dimensionality: {cfg.dim}')

save_folder = f'{cfg.root_data_folder}/{cfg.name}'

os.makedirs(save_folder, exist_ok=True)

if os.path.isfile(f'{save_folder}/config.py'):
    if not filecmp.cmp(args.config_path, f'{save_folder}/config.py', shallow=False):
        raise Exception('A different config found!')
else:
    shutil.copy2(args.config_path, f'{save_folder}/config.py')

load_random(f'{save_folder}/random_state.pkl', cfg.random_seed)


def gen_initial_data():
    data = uni_sample(cfg.domain, cfg.n_queries_initial)
    if cfg.put_corners:
        avg = 0.5 * (cfg.function.domain.lower_limit_vector + cfg.function.domain.upper_limit_vector)
        corner_data = utils.generate_grid([np.array([cfg.function.domain.lower_limit_vector[i],
                                                     cfg.function.domain.upper_limit_vector[i]])
                                           for i in range(cfg.dim)], cfg.dim)
        corner_data = corner_data * 0.99 + 0.01 * np.random.random(size=corner_data.shape) * avg[None, :]
        data = np.concatenate((corner_data, data), axis=0)
    return data


data = try_load(f'{save_folder}/annr_data.npy', gen_initial_data)

save_indices = np.array([100, 1000, 10000, 100000, 1000000, 10000000]).astype(int)

values = try_load(f'{save_folder}/afi_values.npy',
                  lambda: np.array([cfg.function(x) for x in data], dtype=np.float128))

graph = annr.VoronoiGraph(cfg.strategy)
if cfg.bounds_intersect:
    graph.initialize(data, annr.Unbounded())    # consider all circumcenters
    # graph.initialize(data, annr.BoundingBox(cfg.dim, 10))
else:
    graph.initialize(data, cfg.domain)        # ignore cicrumcenters outside the domain

interpolator = annr.ActiveInterpolator(np.random.randint(0, 1000, dtype=int), graph, values, cfg.criterion)

# generate first spots inside the convex hull
tmp = np.random.uniform(size=(cfg.tape_size, data.shape[0]))
tmp = tmp / np.sum(tmp, axis=1, keepdims=True)
hot_spots = np.array([np.sum(data * a[:, None], axis=0) for a in tmp])

for i in trange(data.shape[0], cfg.n_queries):
    if i in save_indices:
        # Saving intermediate
        np.save(f'{save_folder}/annr_data.npy', graph.get_data())
        np.save(f'{save_folder}/annr_values.npy', interpolator.get_values())
        pickle.dump(np.random.get_state(), open(f'{save_folder}/random_state.pkl', 'wb'))

    time0 = time()
    vertex, barycenters = interpolator.search_simplex(
        hot_spots, cfg.num_random_steps, cfg.tape_size)
    hot_spots = barycenters

    time1 = time()

    simplex = vertex.dual
    query = vertex.ref

    if cfg.bounds_intersect and not cfg.domain.contains(query):
        # intersect [barycenter; query] with the bounds
        ref = barycenters[0]
        u = query - ref
        u /= np.linalg.norm(u)
        query = ref + u * cfg.domain.max_length(ref, u)

    assert np.sum(~np.isfinite(query)) == 0, f'Bad query: {query}'

    time2 = time()

    value = cfg.function(query)

    time3 = time()

    interpolator.insert_point(query, value)
    time4 = time()

    # print(f'Times:\n'
    #       f'  search_simplex: {time1 - time0}\n'
    #       f'  function query: {time3 - time2}\n'
    #       f'  pnt. insertion: {time4 - time3}')

np.save(f'{save_folder}/annr_data.npy', graph.get_data())
np.save(f'{save_folder}/annr_values.npy', interpolator.get_values())
pickle.dump(np.random.get_state(), open(f'{save_folder}/random_state.pkl', 'wb'))

