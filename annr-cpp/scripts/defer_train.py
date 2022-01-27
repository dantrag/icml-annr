import argparse
parser = argparse.ArgumentParser()
parser.add_argument('config_path', help='a configuration file path')
args = parser.parse_args()

import utils
cfg = utils.load_config(args.config_path)

import pickle
import shutil
import filecmp
import os
import numpy as np

from density_functions.gravitational_wave import *
from density_functions.misc import *
from defer.helpers import *
from defer.variables import Variable
from defer.bounded_space import sample_uniform

print(f'Function: {cfg.function.name}')
print(f'Bounds: {cfg.domain}')
print(f'Space dimensionality: {cfg.dim}')

save_folder = f'{cfg.root_data_folder}/{cfg.name}'

os.makedirs(save_folder, exist_ok=True)
os.makedirs(f'{save_folder}/extra', exist_ok=True)

if os.path.isfile(f'{save_folder}/config.py'):
    if not filecmp.cmp(args.config_path, f'{save_folder}/config.py', shallow=False):
        raise Exception(f'A different config found in {save_folder}!')
else:
    shutil.copy2(args.config_path, f'{save_folder}/config.py')

np.random.seed(cfg.random_seed + 814)

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
    num_fn_calls=cfg.n_queries,
    callback=lambda i, density:
    print("#Evals: %s. Log Z: %.2f" %
          (density.num_partitions, np.log(density.z))),
    callback_freq_fn_calls=max(cfg.n_queries // 1000, 1),
    is_vectorized_fn=False,
    savepoint_callback=lambda density, i:
        density.save(f'{save_folder}/extra/{i}_defer.pkl')
)

# pickle.dump(approx, open(f'{save_folder}/defer.pkl', 'wb'))
approx.save(f'{save_folder}/defer.pkl')
