import sys
sys.path.append('../../')

from dafaq.function import ImageFunctionBinary
from dafaq.interpolator import DelaunayInterpolatorBoundaryIntersect as ANNR,\
                                PartitioningInterpolator as DEFER

### Load function as 2D image
function = ImageFunctionBinary('spiral.png')

### Default domain of an image-loaded function is [0, H-1] x [0, W-1]
domain = function.default_domain()

interpolators = [
#    ('DEFER', DEFER(function, domain)),
#    ('ANNR', ANNR(function, domain)),
]


for name, interpolator in interpolators:
    for n in [200, 400]:
        score = interpolator.run(n)
        interpolator.save_plot(title=f'{name}, {n} iterations',
                               show_axes=False, highlight_seeds=False)


### Save a ground truth data plot
function.save_plot(filename=f'{function.name}_ground_truth.png',
                   domain=domain,
                   title=rf'Ground-truth values',
                   resolution=500, use_tex=True, show_axes=False)
