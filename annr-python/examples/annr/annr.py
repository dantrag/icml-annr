import sys
sys.path.append('../../')

from pyannr.function import ImageFunction
from pyannr.interpolator import DelaunayInterpolatorBoundaryIntersect as ANNR,\
                                PartitioningInterpolator as DEFER

### Load function as 2D image
function = ImageFunction('annr-blur.png', invert=True)

### Default domain of an image-loaded function is [0, H-1] x [0, W-1]
domain = function.default_domain()

interpolators = [
    ('ANNR', ANNR(function, domain)),
]


for name, interpolator in interpolators:
    for i in range(20):
        score = interpolator.run(100, continues=True)
        interpolator.save_plot(filename=f'annr_{(i + 1) * 100}',
                               title='',
                               show_axes=False, highlight_seeds=False,
                               delauney=False, partitioning=True, points=False)

