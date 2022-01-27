import sys
sys.path.append('../../')

from pyannr.domain import RectangularDomain
from pyannr.function import L1Norm
from pyannr.interpolator import DelaunayInterpolatorBoundaryIntersect as ANNR,\
                               PartitioningInterpolator as DEFER

function = L1Norm()
domain = RectangularDomain([-1, -1], [1, 1])

interpolators = [
    ('DEFER', DEFER(function, domain)),
    ('ANNR', ANNR(function, domain)),
    ('ANNR', ANNR(function, domain)),
    ('ANNR', ANNR(function, domain)),
    ('ANNR', ANNR(function, domain)),
    ('ANNR', ANNR(function, domain)),
]

### Setting up evaluation points and ground truth data for faster evaluation
import numpy as np
test_set = domain.grid(300)
test_values = np.array([function(x) for x in test_set])
MAE = lambda x, yhat: np.average(abs(yhat - test_values))


scores = []
for name, interpolator in interpolators:
        score = interpolator.run(1000, evaluate=True, evaluation_frequency=50,
                                 evaluation_set=test_set, evaluation_metrics=[MAE])
        scores.append((score, name))
        interpolator.save_plot(title=f'{name}, 1000 iterations',
                               show_axes=False, highlight_seeds=False)
print(scores)

### Save score graphs

from pyannr.utils import save_score_plot
save_score_plot(scores,
                filename=f'{function.name}_mae_scores_300x300test.png',
                title=rf'MAE scores',
                metric_names=['MAE'], skip_first=2,
                tex=True, log_scale=False)

### Save a ground truth data plot

function.save_plot(filename=f'{function.name}_ground_truth.png',
                   domain=domain,
                   title=rf'Ground-truth values',
                   resolution=500, use_tex=False, show_axes=False)