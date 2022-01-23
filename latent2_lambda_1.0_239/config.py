afi_path = 'cmake-build-conda/vgt/'
defer_path = 'approximator/defer/'

import sys
sys.path.append(afi_path)
sys.path.append(defer_path)

import afi
from density_functions.misc import *
from density_functions.gravitational_wave import *
from utils import *
import numpy as np

dim = 2

import vae_vf
# Function to query
# function = normalize_function(as_float128(load_gw_injected()))[0]
function = f2sim(f'latent{dim}', vae_vf.load_volume_form(), dim, -1., 1.)
# function = load_gaussian(dim, 10)
# function = f2sim('sphere6', lambda x: 1 if np.linalg.norm(x) < 1 else 0, dim, -2., 2.)


# Bounds, defaults to afi.BoundingBox(function.domain)
# Options: afi.BoundingBox(int dim, float half_width)
#          afi.BoundingBox(vec<float> lower, vec<float> upper)
#          afi.BoundingSphere(int dim, float radius)
domain = None

# Desired number of queries
n_queries = 1000

# Simplex scoring
lmbda = 1.
criterion = afi.CayleyMengerCriterion(lmbda)

# Initial data size
n_queries_initial = 10

# How many walkers simultaneously
tape_size = 50

# How many steps in one walk
num_random_steps = 5

# Add corners of the bounding box at the start
put_corners = True

# Intersect [circumcenter, barycenter] with the boundary when the former is outside if True
# Ignore circumcenters outside of the bounds if False
bounds_intersect = True

# Strategy alternatives: BIN_SEARCH, BRUTEFORCE_GPU
strategy = afi.RayStrategyType.BIN_SEARCH

random_seed = 239
# uniform_data_seed = 781
test_seed = 963

root_data_folder = 'data/afi'
# Folder name
name = f'{function.name}_lambda_{lmbda}_{random_seed}'


################## DEFAULTS ##################
if domain is None:
    domain = afi.BoundingBox(function.domain.lower_limit_vector,
                             function.domain.upper_limit_vector)
assert dim == function.domain.lower_limit_vector.size
_version = '1'
