annr_path = 'build/annr/'
defer_path = 'defer/'

import sys
sys.path = [annr_path, defer_path] + sys.path

import annr
from density_functions.misc import *
from density_functions.gravitational_wave import *
from utils import *
import numpy as np

dim = 6

_e_lambda = 8000
# Function to query
# function = normalize_function(as_float128(load_gw_injected()))[0]
function = unlog_function(normalize_function(as_float128(load_gw_injected()))[0], _e_lambda)
# function = load_gaussian(dim, 10)
# function = f2sim('sphere6', lambda x: 1 if np.linalg.norm(x) < 1 else 0, dim, -2., 2.)


# Bounds, defaults to annr.BoundingBox(function.domain)
# Options: annr.BoundingBox(int dim, float half_width)
#          annr.BoundingBox(vec<float> lower, vec<float> upper)
#          annr.BoundingSphere(int dim, float radius)
domain = None

# Desired number of queries
n_queries = 100000

# Simplex scoring
lmbda = float(${_LAMBDA})
angle = "${_ANGLE}"
if angle == "" or angle == "None" or angle == "none":
    criterion = annr.CayleyMengerCriterion(lmbda)
else:
    criterion = annr.ClippedCayleyMengerCriterion(lmbda, float(angle))

# Initial data size
n_queries_initial = 30

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
strategy = annr.RayStrategyType.BIN_SEARCH

random_seed = ${_RS}
test_seed = 781

root_data_folder = 'data'
# Folder name
name = f'{function.name}_elambda_{_e_lambda}_lambda_{lmbda}_angle_{angle}_{random_seed}'
# assert lmbda == 1


################## DEFAULTS ##################
if domain is None:
    domain = annr.BoundingBox(function.domain.lower_limit_vector,
                             function.domain.upper_limit_vector)
assert dim == function.domain.lower_limit_vector.size
_version = '1'
