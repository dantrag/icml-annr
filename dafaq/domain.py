from abc import ABCMeta, abstractmethod
import numpy as np
from numpy.random import random

# To define your own domain, derive from Domain and implement the abstract
# methods. If it is a trivial task, implement intersection with a segment,
# otherwise it will be done approximately via binary search

class Domain(metaclass=ABCMeta):
    @abstractmethod
    def contains(self, point: np.ndarray):
        raise NotImplementedError
        return False

    @abstractmethod
    def bbox(self):
        raise NotImplementedError
        return None

    def bbox_dimensions(self):
        return self.bbox()[1] - self.bbox()[0]

    # Intersect the domain boundary with a segment between point_0 and point_1
    def intersect(self, point0, point1):
        assert self.contains(point0) != self.contains(point1),\
               f"Exactly one of the points ({point0}, {point1}) must be inside the domain boundary!"
        if self.contains(point1):
            point0, point1 = point1, point0

        # From now on, point0 is inside the boundary, and point1 is outside
        # Perform a binary search to intersect with the boundary
        left = 0.0
        right = 1.0
        while right - left > 0.00001:
            mid = (left + right) / 2
            point = point0 * (1 - mid) + point1 * mid
            if self.contains(point):
                left = mid
            else:
                right = mid

        # Return the result that is on the safe side (inside) of the boundary
        return point0 * (1 - left) + point1 * left

    def dimensionality(self):
        return self.d

    def grid(self, resolution):
        if np.isscalar(resolution):
            resolution = [resolution] * self.d
        assert len(resolution) == self.d,\
               f"Dimensionality of the domain ({self.d}) does not match the one of the supplied grid resolution ({len(resolution)})"
        axes = []
        bbox = self.bbox()
        for i in range(self.d):
            axes.append(np.linspace(bbox[0][i], bbox[1][i], resolution[i]))
        axes = np.meshgrid(*axes)
        axes = tuple([np.reshape(axis, (-1, 1)) for axis in axes])
        return np.hstack(axes)

    def corners(self):
        points = []
        bbox = self.bbox()
        for mask in range(1 << self.d):
            point = []
            for i in range(self.d):
                point.append(bbox[bool(mask & (1 << i))][i])
            points.append(np.array(point))
        return np.array(points)

    def sample_random(self, count):
        samples = []
        lower, upper = self.bbox()
        attempts = 1000
        while count and attempts:
            point = random(self.d) * (upper - lower) + lower
            if self.contains(point):
                count -= 1
                samples.append(point)
            else:
                attempts -= 1
        return np.array(samples)


class RectangularDomain(Domain):
    def __init__(self, lower, upper):
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        assert self.lower.shape == self.upper.shape,\
               f"Lower and upper limits of the rectangular domain should have the same shape\
                 ({self.lower.shape} != {self.upper.shape})"
        self.lower, self.upper = np.minimum(self.lower, self.upper), np.maximum(self.lower, self.upper)
        self.d = len(self.lower)

    def bbox(self):
        return self.lower, self.upper

    def contains(self, point, precision=0.00001):
        point = np.array(point)
        assert point.shape == self.lower.shape,\
               f"Shape of the point ({point.shape}) does not match the domain limits shape ({self.lower.shape})"

        margin = np.ones(point.shape[0]) * precision
        return (self.lower - margin <= point).all() and (point <= self.upper + margin).all()

    def intersect(self, point0, point1):
        assert self.contains(point0) != self.contains(point1),\
               f"Exactly one of the points ({point0}, {point1}) must be inside the domain boundary!"
        if self.contains(point1):
            point0, point1 = point1, point0

        # From now on, point0 is inside the boundary, and point1 is outside

        for i in range(self.d):
            for face_x in [self.lower[i], self.upper[i]]:
                # segment endpoints should be on different sides on d axis
                if (point0[i] - face_x) * (point1[i] - face_x) < 0:
                    ratio = (face_x - point0[i]) / (point1[i] - point0[i])
                    intersection = [ratio * (point1[j] - point0[j]) + point0[j] for j in range(self.d)]
                    if self.contains(intersection):
                        return intersection

        assert False, f"Did not find an intersection with ({point0}, {point1})"

    @classmethod
    def unit(cls, dimensions=2):
        return RectangularDomain(np.zeros(dimensions), np.ones(dimensions))

class SphericalDomain(Domain):
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius
        self.d = len(self.center)
        assert np.isscalar(self.radius),\
               f"Radius must be a scalar (now {type(self.radius)})"

    def bbox(self):
        unit = np.array([self.radius] * len(self.center))
        return self.center - unit, self.center + unit

    def contains(self, point):
        point = np.array(point)
        assert point.shape == self.center.shape,\
               f"Shape of the point ({point.shape}) does not match the domain limits shape ({self.center.shape})"
        return sum((point - self.center) ** 2) <= self.radius ** 2

    @classmethod
    def unit(cls, dimensions=2):
        return SphericalDomain(np.zeros(dimensions), 1.0)
