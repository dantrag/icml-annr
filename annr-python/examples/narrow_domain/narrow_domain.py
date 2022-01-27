import sys
sys.path.append('../../')

import math
import numpy as np
from pyannr.function import Function
from pyannr.domain import Domain
from pyannr.interpolator import DelaunayInterpolatorBoundaryIntersect as ANNR,\
                               PartitioningInterpolator as DEFER

### Signed distance to a boundary of balls intersection
### balls: (-3, -3), 5 and (4, 4), 5
def distance(x):
    distance1 = 5 - math.sqrt(np.sum((x - np.array([-3, -3])) ** 2))
    distance2 = 5 - math.sqrt(np.sum((x - np.array([4, 4])) ** 2))
    return min(distance1, distance2)

class CustomDomain(Domain):
    d = 2

    def contains(self, point: np.ndarray):
        return distance(point) >= 0

    def bbox(self):
        return np.array([0, 0]), np.array([1, 1])

class CustomFunction(Function):
    name = "distance_to_domain_boundary"

    def __call__(self, x):
        # value outside the domain is 0
        return max(distance(x), 0)

function = CustomFunction()
domain = CustomDomain()

interpolators = [
    ('DEFER', DEFER(function, domain)),
    ('ANNR', ANNR(function, domain, domain.sample_random(10))),
    ('ANNR', ANNR(function, domain, domain.sample_random(10))),
    ('ANNR', ANNR(function, domain, domain.sample_random(10))),
    ('ANNR', ANNR(function, domain, domain.sample_random(10))),
    ('ANNR', ANNR(function, domain, domain.sample_random(10))),
]

### Setting up evaluation points and ground truth data for faster evaluation
test_set = domain.grid(100)
test_set = np.array([x for x in test_set if domain.contains(x)])
test_values = np.array([function(x) for x in test_set])
MAE = lambda x, yhat: np.average(abs(yhat - test_values))

scores = []
for name, interpolator in interpolators:
        score = interpolator.run(1000, evaluate=True, evaluation_frequency=50,
                                 evaluation_set=test_set, evaluation_metrics=[MAE])
        scores.append((score, name))
        interpolator.save_plot(title=f'{name}, 1000 iterations',
                               show_axes=False, highlight_seeds=False)


#function.save_plot('gt.png', domain, 'Ground-truth values')

### Save score graphs

from pyannr.utils import save_score_plot
save_score_plot(scores,
                filename=f'{function.name}_mae_scores.png',
                title=rf'MAE scores',
                metric_names=['MAE'], skip_first=2,
                tex=True, log_scale=False)