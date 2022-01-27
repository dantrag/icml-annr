annr_path = 'build/annr/'
defer_path = 'defer/'

import sys
sys.path.append(annr_path)
sys.path.append(defer_path)

import annr
from density_functions.misc import *
from density_functions.gravitational_wave import *
from utils import *
import numpy as np

dim = 6
_hw = 2  # 2 ** (1/dim)

# Function to query
# function = normalize_function(load_gw_injected())[0]
# function = unlog_function(normalize_function(load_gw_injected())[0])
# function = load_gaussian(dim, 10)
function = f2sim(f'ball{dim}',
                 lambda x: 1 if np.linalg.norm(x) < 1 else 0, dim, -_hw, _hw)


# Bounds, defaults to annr.BoundingBox(function.domain)
# Options: annr.BoundingBox(int dim, float half_width)
#          annr.BoundingBox(vec<float> lower, vec<float> upper)
#          annr.BoundingSphere(int dim, float radius)
# domain = annr.BoundingSphere(dim, 2 ** (1/dim))
domain = annr.BoundingBox(dim, _hw)

# Desired number of queries
n_queries = 1000

# Simplex scoring
lmbda = 10.
criterion = annr.CayleyMengerCriterion(lmbda)

# Initial data size
n_queries_initial = 50

# How many walkers simultaneously
tape_size = 50

# How many steps in one walk
num_random_steps = 5

# Add corners of the bounding box at the start
put_corners = False

# Intersect [circumcenter, barycenter] with the boundary when the former is outside if True
# Ignore circumcenters outside of the bounds if False
bounds_intersect = True

# Strategy alternatives: BIN_SEARCH, BRUTE_FORCE_GPU
strategy = annr.RayStrategyType.BIN_SEARCH

random_seed = 239

root_data_folder = 'data'
# Folder name
name = f'{function.name}_lambda_{lmbda}'

################## DEFAULTS ##################
if domain is None:
    domain = annr.BoundingBox(function.domain.lower_limit_vector,
                              function.domain.upper_limit_vector)
assert dim == function.domain.lower_limit_vector.size
_version = '1'
