from dafaq.function import *
from dafaq.domain import *
from dafaq.interpolator import *
from dafaq.utils import *

from scipy import stats

### Setting up the function and the domain

domain = RectangularDomain([-1, -1], [1, 1])
#domain = RectangularDomain([-1.5, -1.5], [1.5, 1.5])
#function = RotatedFunction(Ellipse([0, 0], [1, 0.5]), 30)
#ellipse.save_plot("ellipse.png", domain, "Ground truth", resolution=300)

#function = NormalGaussian([0, 0], 0.1)
#function = Rosenbrock(1, 100)
#function = Himmelblau()
#function = L2Norm()
#function = Rastrigin()
#function = ImageFunctionBinary("spiral.png", RectangularDomain.unit())
function = StyblinskiTang()
#domain = RectangularDomain.unit()
domain = function.default_domain()
#function.save_plot("gaussian/ground-truth.png", domain, rf"Ground truth, ${function.equation}$", resolution=500, show_axes=False)

### Setting up evaluation points and ground truth data for faster evaluation

test_set = domain.grid(100)
test_values = np.array([function(x) for x in test_set])
MAE = lambda x, yhat: np.average(abs(yhat - test_values))
MAD = lambda x, yhat: np.median(abs(yhat - test_values))

### Random seed generator for ANNR

def random_seeds():
    seeds = domain.sample_random(10)
    corners = domain.corners()
    seeds = np.concatenate((seeds, corners), axis=0)
    return seeds

### Interpolators for the experiments

interpolators = [
    ('DEFER', PartitioningInterpolator(function, domain)),
#    ('ANNR L=0.0001', DelaunayInterpolatorBoundaryIntersect(function, domain, random_seeds(), Lambda=0.0001)),
#    ('ANNR L=0.001', DelaunayInterpolatorBoundaryIntersect(function, domain, random_seeds(), Lambda=0.001)),
#    ('ANNR L=0.01', DelaunayInterpolatorBoundaryIntersect(function, domain, random_seeds(), Lambda=0.01)),
#    ('ANNR L=0.1', DelaunayInterpolatorBoundaryIntersect(function, domain, random_seeds(), Lambda=0.1)),
#    ('ANNR L=1', DelaunayInterpolatorBoundaryIntersect(function, domain, random_seeds(), Lambda=1)),
#    ('ANNR', DelaunayInterpolatorBoundaryIntersect(function, domain, random_seeds(), Lambda=1)),
#    ('ANNR', DelaunayInterpolatorBoundaryIntersect(function, domain, random_seeds(), Lambda=1)),
#    ('ANNR L=10', DelaunayInterpolatorBoundaryIntersect(function, domain, random_seeds(), Lambda=10)),
#    ('ANNR L=100', DelaunayInterpolatorBoundaryIntersect(function, domain, random_seeds(), Lambda=100)),
#    ('ANNR L=1/890', DelaunayInterpolatorBoundaryIntersect(function, domain, random_seeds(), Lambda=890)),
#    ('ANNR L=1000', DelaunayInterpolatorBoundaryIntersect(function, domain, random_seeds(), Lambda=1000)),
]

scores = []
for name, interpolator in interpolators:
    score = interpolator.run(1000, evaluate=False, evaluation_frequency=100,
                             evaluation_set=test_set, evaluation_metrics=[MAE])
    #interpolator.save_plot(f'MAE: {score[-1][1][0]:.3g}')
    interpolator.save_plot(title='', delaunay=False, show_axes=False, highlight_seeds=False)
    scores.append((score, name))

#np.save("gaussian/L=0.1-100_500.npy", scores)
#scores = np.load("gaussian/L=0.1-100_500.npy", allow_pickle=True)

#ours = DelaunayInterpolatorBoundaryIntersect(function, RectangularDomain.unit(), [])
#ours.load_points_from_file('spiral/dafaq_image_0x7fe24952e930_400.npy')
#ours.save_plot(title='', delaunay=False, show_axes=False, highlight_seeds=False)


#import sys
#sys.exit(0)

### Saving the scores plot (install texlive-latex-extra texlive-fonts-recommended dvipng cm-super to use latex or disable it in utils.load_plot_style)

save_score_plot(scores[::-1], metric_names=['MAE'],
                filename='ST_500.png', title=rf'Approximation of $f(x, y) = {function.equation}$', skip_first=1, tex=True, log_scale=False)