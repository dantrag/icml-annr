import sys
sys.path.append('../../')

from pyannr.function import ImageFunctionBinary
from pyannr.interpolator import DelaunayInterpolatorBoundaryIntersect as ANNR,\
                                PartitioningInterpolator as DEFER

### Load function as 2D image
function = ImageFunctionBinary('spiral.png')

### Default domain of an image-loaded function is [0, H-1] x [0, W-1]
domain = function.default_domain()

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
test_set = domain.grid(100)
test_values = np.array([function(x) for x in test_set])
MAE = lambda x, yhat: np.average(abs(yhat - test_values))


scores = []
for name, interpolator in interpolators:
        score = interpolator.run(1000, evaluate=True, evaluation_frequency=50,
                                 evaluation_set=test_set, evaluation_metrics=[MAE])
        scores.append((score, name))
print(scores)

### Save score graphs

from pyannr.utils import save_score_plot
save_score_plot(scores,
                filename=f'{function.name}_mae_scores.png',
                title=rf'MAE scores',
                metric_names=['MAE'], skip_first=2,
                tex=True, log_scale=False)