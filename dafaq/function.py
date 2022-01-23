import math
import numpy as np
import matplotlib.pyplot as plt

from abc import ABCMeta, abstractmethod
from functools import lru_cache
from pathlib import Path

from dafaq.utils import load_plot_style, reset_plot_style

from .domain import Domain, RectangularDomain


class Function:
    def __call__(self, x):
        raise NotImplementedError("Function is a base class, derive it and define your own function")

    name = "undefined"
    equation = "undefined"

    @classmethod
    def default_domain(cls, dimensions=2):
        return RectangularDomain(np.zeros(dimensions), np.ones(dimensions))

    def draw(self, axes, domain, resolution=300, min_value=None, max_value=None, **args):
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
                        vmax=(max_value if max_value else np.max(values)), # artificially tone down
                        interpolation='nearest',
                        extent=(bbox[0][0],bbox[1][0], bbox[0][1], bbox[1][1]),
                        alpha=0.9)


    def save_plot(self, filename: str, domain, title="", resolution=300, use_tex=True, **args):
        load_plot_style(use_tex)

        figure, axes = plt.subplots()
        self.draw(axes, domain, resolution, **args)
        axes.set_title(title)
        if 'show_axes' in args:
            if not args['show_axes']:
                axes.set_axis_off()

        figure.tight_layout()
        figure.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.05)

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
    equation = r"e^{-\|x\|_2^2}"

    def __init__(self, A=1, B=1):
        self.A = A
        self.B = B
        self.equation = (f"{self.A:.3g}" if self.A != 1 else "") + "e^{-" + (f"{self.B:.3g}" if self.B != 1 else "") + "x^2}"

    def __call__(self, x):
        return self.A * math.exp(- self.B * sum(np.array(x) ** 2))


class NormalGaussian(Function):
    name = "gaussian"

    def __init__(self, mu, sigma_squared):
        self.mu = np.array(mu)
        self.sigma_squared = sigma_squared
        if not np.isscalar(sigma_squared):
            raise NotImplementedError("Only scalar sigma is implemented!")
        if sigma_squared < 0:
            raise ValueError("Squared sigma (variance) should be positive!")
        self.equation = r"\frac{1}{\sqrt{2\pi" + (rf"\cdot{self.sigma_squared:.3g}" if self.sigma_squared != 1 else "") + r"}}" +\
                        r"\exp\left(-\frac{\|x" + (rf"-{str(self.mu)}" if (self.mu != np.zeros(len(self.mu))).all() else "") +\
                        r"\|^2}{2" + (rf"\cdot{self.sigma_squared:.3g}" if self.sigma_squared != 1 else "") + r"}\right)"

    def __call__(self, x):
        return 1 / math.sqrt(2 * math.pi * self.sigma_squared) * math.exp(- sum((np.array(x) - self.mu) ** 2) / 2 / self.sigma_squared)


class L1Norm(Function):
    name = "L1-norm"
    equation = r"\|x\|_1"

    @classmethod
    def __call__(cls, x):
        return sum(abs(x))

class L2Norm(Function):
    name = "L2-norm"
    equation = r"\|x\|_2"

    @classmethod
    def __call__(cls, x):
        return math.sqrt(sum(x ** 2))

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
    equation = "(x^2 + y - 11)^2 + (x + y^2 - 7)^2"

    @classmethod
    def __call__(cls, x):
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

    @classmethod
    def default_domain(cls):
        return RectangularDomain([-5, -5], [5, 5])

class HimmelblauLike(Function):
    name = "himmelblau-like"

    @classmethod
    def __call__(cls, x):
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) + 12

class Rosenbrock(Function):
    name = "rosenbrock"

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.equation = "(" + (f"{self.a:.3g}" if self.a else "") + " - x)^2 + " +\
                        (f"{self.b:.3g}" if self.b != 1 else "") + "(y - x^2)^2"

    def __call__(self, x):
        return (self.a - x[0]) ** 2 + self.b * (x[1] - x[0] ** 2) ** 2

class Rastrigin(Function):
    name = "rastrigin"

    def __init__(self, a=10):
        self.a = a
        self.equation = (f"{self.a:.3g}" if self.a != 1 else "") + r"\dim(x) + \Sigma\left[x_i^2 - " +\
                        (f"{self.a:.3g}" if self.a != 1 else "") + r"\cos{2\pi x_i}\right]"

    def __call__(self, x):
        return self.a * len(x) + sum(x ** 2 - np.cos(x * 2 * math.pi))

    @classmethod
    def default_domain(cls, dimensions=2):
        return RectangularDomain(np.ones(dimensions) * (-5.12),
                                 np.ones(dimensions) * 5.12)

class StyblinskiTang(Function):
    name = "styblinski-tang"

    def __init__(self, dimensions=2):
        self.d = dimensions

    @classmethod
    def __call__(cls, x):
        return 0.5 * np.sum(x**4 - 16 * x ** 2 + 5 * x) + 80

    def default_domain(self):
        return RectangularDomain(np.ones(self.d) * (-5), np.ones(self.d) * 5)

### Functions from images (R^2 -> [0, 1])

class ImageFunctionBinary(Function):
    name = ""

    def __init__(self, filename, domain=None):
        self.name = Path(filename).stem
        self.pixels = np.array(plt.imread(filename))
        # convert to grayscale (ignore alpha)
        self.pixels = np.dot(self.pixels[..., :3], np.ones(3) / 3)
        self.dimensions = self.pixels.shape[:2]

        if not domain:
            domain = RectangularDomain(np.zeros(2), self.pixels.shape - np.ones(2))
        assert domain.dimensionality() == 2, "A function loaded from an image must have a 2-dimensional domain!"
        self.domain = domain
        self.bbox = self.domain.bbox()

    def __call__(self, x):
        x = np.asarray(x)
        assert len(x) == 2, f"Query dimension must be 2, not {len(x)} for a function loaded from an image"
        # transform query coodinates to unit cube and then to image coordinates
        x = (x - self.bbox[0]) / (self.bbox[1] - self.bbox[0])
        x = x * (self.dimensions - np.ones(2)) + np.ones(2) * 0.5
        x = x.astype(int)
        x = np.minimum(x, self.bbox[1])
        x = np.maximum(x, self.bbox[0])
        x = x.astype(int)

        return float(self.pixels[x[0]][x[1]] > 0)

    def default_domain(self):
        return self.domain

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



