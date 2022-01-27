from dafaq.function import NormalGaussian, Ellipse
from dafaq.domain import *
from dafaq.utils import *
from dafaq.interpolator import DelaunayInterpolatorBoundaryIntersect as ANNR,\
                               PartitioningInterpolator as DEFER

from scipy import stats


### Setting up the function and the domain

domain = RectangularDomain([-2, -2], [2, 2])
#function = NormalGaussian([0, 0])
function = Ellipse([0, 0], [1, 1.5])


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
    ('DEFER', DEFER(function, domain)),
    ('ANNR', ANNR(function, domain, random_seeds())),
]


scores = []
for name, interpolator in interpolators:
    score = interpolator.run(300, evaluate=True, evaluation_frequency=50,
                             evaluation_set=test_set, evaluation_metrics=[MAE])
    interpolator.save_plot(title=f'MAE: {score[-1][1][0]:.3g}',
                           delaunay=False, show_axes=False, highlight_seeds=False)
    scores.append((score, name))
# print(scores)


### Plotting the function ground truth (install texlive-latex-extra
# texlive-fonts-recommended dvipng cm-super
# to use latex or disable it in utils.load_plot_style)

function.save_plot(filename=f"{function.name}_ground_truth.png",
                   title=rf"Ground truth, ${function.equation}$",
                   domain=domain,
                   resolution=500, use_tex=True, show_axes=False)

### Saving the scores plot

save_score_plot(scores[::-1],
                filename='ST_500.png',
                title=rf'Approximation of $f(x, y) = {function.equation}$',
                metric_names=['MAE'], skip_first=1,
                tex=True, log_scale=False) 
