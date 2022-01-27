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

### Special type of score function -- number of connected components
from scipy.ndimage.measurements import label
connection = np.ones((3, 3), dtype=np.int)
NCC = lambda x, yhat: label(np.reshape(np.array(yhat, dtype=np.int), (100, 100)),
                            connection)[1]


scores = []
for name, interpolator in interpolators:
        score = interpolator.run(1000, evaluate=True, evaluation_frequency=20,
                                 evaluation_set=test_set, evaluation_metrics=[NCC])
        print(score)
        scores.append((score, name))
print(scores)


### Save score graphs

from pyannr.utils import save_score_plot
save_score_plot(scores,
                filename=f'{function.name}_ncc_scores.png',
                title=f'Number of connected components',
                metric_names=[''], skip_first=1,
                tex=True, log_scale=False)
