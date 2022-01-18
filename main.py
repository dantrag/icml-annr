from dafaq.function import *
from dafaq.domain import *
from dafaq.interpolator import *
from dafaq.utils import *

### Setting up the function and the domain

domain = RectangularDomain([-3, -3], [3, 3])
#domain = RectangularDomain([-1.5, -1.5], [1.5, 1.5])
#ellipse = RotatedFunction(Ellipse([0, 0], [1, 0.5]), 20)
#ellipse.save_plot("ellipse.png", domain, "Ground truth", resolution=300)

function = Gaussian(1, 0.5)

### Setting up evaluation points and ground truth data for faster evaluation

test_set = domain.grid(100)
test_values = np.array([function(x) for x in test_set])
MAE = lambda x, yhat: np.average(abs(yhat - test_values))

### Random seed generator for ANNR

def random_seeds():
    seeds = domain.sample_random(10)
    corners = domain.corners()
    seeds = np.concatenate((seeds, corners), axis=0)
    return seeds

### Interpolators for the experiments

interpolators = [
    ('DEFER', PartitioningInterpolator(function, domain)),
    ('ANNR', DelaunayInterpolatorBoundaryIntersect(function, domain, random_seeds(), Lambda=3)),
    ('ANNR', DelaunayInterpolatorBoundaryIntersect(function, domain, random_seeds(), Lambda=3)),
    ('ANNR', DelaunayInterpolatorBoundaryIntersect(function, domain, random_seeds(), Lambda=3)),
]

scores = []
for name, interpolator in interpolators:
    score = interpolator.run(50, evaluate=True, evaluation_frequency=25,
                             evaluation_set=test_set, evaluation_metrics=[MAE])
    interpolator.save_plot(f'MAE: {score[-1][1][0]:.3g}')
    scores.append((score, name))

### Saving the scores plot (install texlive-latex-extra texlive-fonts-recommended dvipng cm-super to use latex or disable it in utils.load_plot_style)

save_score_plot(scores[::-1], metric_names=['MAE'],
                filename='test.png', title=r'$e^{-x^2/2}$ approximation', skip_first=1, tex=True)