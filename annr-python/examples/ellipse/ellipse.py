import sys
sys.path.append('../../')

from pyannr.domain import RectangularDomain
from pyannr.function import Ellipse, RotatedFunction
from pyannr.interpolator import DelaunayInterpolatorBoundaryIntersect as ANNR,\
                               PartitioningInterpolator as DEFER


domain = RectangularDomain([-1.5, -1.5], [1.5, 1.5])

scores_per_angle = []

### 10 experiments with different rotations
for i in range(10):
    scores_per_angle.append([])

    # Ellipse rotated by 5 * i degrees
    function = RotatedFunction(Ellipse([0, 0], [1, 0.5]), 5 * i)

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


    for name, interpolator in interpolators:
            interpolator.run(300)
            score = interpolator.evaluate_set(test_set, [MAE])
            scores_per_angle[i].append(([5 * i, score], name))
            interpolator.save_plot(title='',
                                   show_axes=False, highlight_seeds=False)


### Save score graphs

scores = [([], name) for score, name in scores_per_angle[0]]
for score_per_angle in scores_per_angle:
    for i in range(len(score_per_angle)):
        scores[i][0].append(score_per_angle[i][0])
print(scores)

from pyannr.utils import save_score_plot
save_score_plot(scores,
                filename=f'ellipse_mae_scores.png',
                title=rf'MAE scores',
                metric_names=['MAE'], skip_first=0,
                tex=True, log_scale=False, x_suffix=r'${}^{\circ}$')
