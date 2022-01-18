from abc import ABCMeta, abstractmethod
import math
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache

from dafaq.utils import load_plot_style, reset_plot_style

from .domain import Domain


class Function:
    def __call__(self, x):
        raise NotImplementedError("Function is a base class, derive it and define your own function")

    name = "undefined"

    def draw(self, axes, domain, resolution=300, min_value=None, max_value=None):
        if domain.dimensionality() != 2:
            raise NotImplementedError("Can only plot on 2-dimensional domains")

        values = np.array([self(x) if domain.contains(x) else 0 for x in domain.grid(resolution)])
        if np.isscalar(resolution):
            resolution = [resolution] * domain.dimensionality()
        bbox = domain.bbox()
        values = values.reshape(tuple(resolution[::-1]))
        values = np.flip(values, axis=0)

        if not isinstance(axes, list):
            axes = [axes]
        for axis in axes:
            axis.imshow(values , cmap='Blues',
                        vmin=(min_value if min_value else np.min(values)),
                        vmax=(max_value if max_value else np.max(values)) * 1.1, # artificially tone down
                        interpolation='nearest',
                        extent=(bbox[0][0],bbox[1][0], bbox[0][1], bbox[1][1]),
                        alpha=0.9)


    def print(self, filename: str, domain, title="", resolution=300):
        load_plot_style()

        figure, axes = plt.subplots(dpi=150)
        self.draw(axes, domain, resolution)
        axes.set_title(title)
        figure.savefig(filename)

        reset_plot_style()


class RotatedFunction(Function):
    def __init__(self, function, angle, dim=2, rotation_plane_dim=[0, 1]):
        assert len(rotation_plane_dim) == 2, "Specify exactly two dimensions for the rotation"
        assert 0 <= max(rotation_plane_dim) < dim, f"Dimension {max(rotation_plane_dim)} is out of bounds(0-{dim - 1})"

        self.function = function

        # Angle in degrees
        self.angle_degrees = angle
        self.angle = angle / 180 * math.pi
        self.rotation_matrix = np.identity(dim)
        dim1, dim2 = rotation_plane_dim
        R = self.rotation_matrix
        R[dim1][dim1] = math.cos(self.angle)
        R[dim2][dim2] = math.cos(self.angle)
        R[dim1][dim2] = -math.sin(self.angle)
        R[dim2][dim1] = math.sin(self.angle)

        self.name = self.function.name + f"_rot_{self.angle_degrees}"

    def __call__(self, x):
        x = self.rotation_matrix.dot(x)
        return self.function(x)


### Primitive functions

# Simple Gaussian in a form of A*exp(-Bx^2)
class Gaussian(Function):
    name = "gaussian"

    def __init__(self, A=1, B=1):
        self.A = A
        self.B = B

    def __call__(self, x):
        return self.A * math.exp(- self.B * sum(np.array(x) ** 2))


class NormalGaussian(Function):
    name = "gaussian"

    def __init__(self, mu, sigma):
        self.mu = np.array(mu)
        self.sigma = sigma
        if hasattr(sigma, "__len__"):
            raise NotImplementedError("Only scalar sigma is implemented!")

    def __call__(self, x):
        return 1 / self.sigma / math.sqrt(2 * math.pi) * math.exp(- sum((np.array(x) - self.mu) ** 2) / 2 / self.sigma ** 2)


class Ellipse(Function):
    name = "ellipse"

    def __init__(self, center, semiaxes):
        assert len(center) == len(semiaxes), f"Center and semiaxes have the same dimensionality ({len(center)} != {len(semiaxes)})"
        assert min(semiaxes) > 0, f"Semiaxes should be positive!"
        self.center = np.array(center)
        self.semiaxes = np.array(semiaxes)

    def __call__(self, x):
        assert len(x) == len(self.center), f"Point should have the same dimensionality ({len(self.center)})"
        return int(sum(((np.array(x) - self.center)/ self.semiaxes) ** 2) <= 1)


### Optimization functions

class Himmelblau(Function):
    name = "himmelblau"

    @classmethod
    def __call__(self, x):
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

class HimmelblauLike(Function):
    name = "himmelblau-like"

    @classmethod
    def __call__(self, x):
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) + 12


### Not upgraded functions

def onlyx(x):
    return x[0]

def yx(x):
    return 1/(abs(x[0] + x[1]) + 1)

def hyperbolic(x):
    return 1/(math.sqrt(x[0]**2 + x[1]**2) + 1)

def hyperbolic_sharp(x):
    return 1/(math.sqrt(x[0]**2 + x[1]**2) * 10 + 1)

def hyperbolic_sharp_inf(x):
    return 1/(math.sqrt(x[0]**2 + x[1]**2) + 0.01)

def cone(x):
    return math.sqrt(x[0]**2 + x[1]**2)

def stripe(x):
    return float(abs(x[0]) < 1)



