from re import S
from dafaq.function import Function
from dafaq.domain import Domain
from dafaq.geometry_utils import *

from abc import ABCMeta, abstractmethod

import numpy as np
import math
from numpy.random import random
from scipy.spatial import Delaunay

import matplotlib.pyplot as plt
from tqdm import tqdm

from dafaq.utils import load_plot_style, reset_plot_style

def adaptive_marker_size(count):
    return min(20, 200 // math.sqrt(count))

class Interpolator(metaclass=ABCMeta):
    name = "undefined"
    base_name = "undefined"

    def __init__(self, function, domain):
        self.function = function
        self.domain = domain

    def plot_points(self):
        raise NotImplementedError

    def plot_partitioning(self):
        raise NotImplementedError

    @abstractmethod
    def sample_count(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, point):
        raise NotImplementedError

    def run(self, iteration_count,
            evaluate=False, evaluation_frequency=1,
            evaluation_set=np.array([]), evaluation_metrics=[]):
        raise NotImplementedError

    def plot(self, axes, points=True, partitioning=True):
        if points:
            self.plot_points(axes)
        if partitioning:
            self.plot_partitioning(axes)

    def save_plot(self, title, points=True, partitioning=True):
        load_plot_style()

        figure, axes = plt.subplots()
        self.plot(axes, points=points, partitioning=partitioning)
        self.function.draw(axes, self.domain)
        axes.set_title(title)
        figure.savefig(f'{self.name}_{self.sample_count()}.png')

        reset_plot_style()
 

class DelaunayInterpolator(Interpolator):
    triangulation = None
    seeds_count = 0
    Lambda = 1.0
    snapshots = []
    function = None
    domain = None
    base_name = "ANNR"
    iteration_offset = 0

    def __init__(self, function, domain, seeds, Lambda=1.0):
        super().__init__(function, domain)

        self.points = np.empty([0, domain.dimensionality()])
        self.seeds_count = len(seeds)
        if len(seeds):
            self.add_points(seeds)
        self.Lambda = Lambda
        self.name = f'dafaq_{function.name}_{hex(id(self))}'
        self.iteration_offset = len(seeds)

    def sample_count(self):
        return len(self.points)

    def add_points(self, new_points):
        self.points = np.concatenate((self.points, new_points), axis=0)
        
        # Recomputing entire triangulation. Can be done more efficiently with Delaunay update algos
        self.triangulation = Delaunay(self.points).simplices
        
        if self.sample_count() % 100 == 0:
            #self.snapshots.append(self.points.copy())
            np.save(f'{self.name}_{self.sample_count()}.npy', self.points)
            #print(len(self.points))

    def score(self, simplex_points):
        vertices = simplex_points.copy()
        f_list = np.array([self.function(x) * self.Lambda for x in vertices])
        lifted_vertices = np.concatenate((vertices, np.expand_dims(f_list, axis=1)), axis=-1)
        return simplex_volume(lifted_vertices)

    def query_new_point(self):
        score_list = [(self.score(self.points[simplex]), simplex)
                      for simplex in self.triangulation if self.domain.contains(circumcenter(self.points[simplex]))]

        # Can be more efficiently implemented with a p-queue (heap)
        best_simplex = max(score_list, key=lambda a : a[0])[1]
        new_point = circumcenter(self.points[best_simplex])

        self.add_points([new_point])

    def run(self, iteration_count,
            evaluate=False, evaluation_frequency=None,
            evaluation_set=np.array([]), evaluation_metrics=[]):      
        print(f"Running {self.base_name} ({self.name})...", flush=True)

        if not evaluate or not evaluation_frequency:
            evaluation_frequency = iteration_count * 2
        score_curve = []

        for iteration in tqdm(range(self.iteration_offset, iteration_count)):
            self.query_new_point()
            if iteration % evaluation_frequency == 0 or\
               iteration == self.iteration_offset or\
               iteration == iteration_count - 1:
                #self.snapshots.append(self.points.copy())
                score_curve.append([iteration,
                                    self.evaluate_set(evaluation_set,
                                                      evaluation_metrics)])
        
        self.iteration_offset = 0

        return score_curve

    def evaluate_set(self, evaluation_set, metrics):
        evaluation_results = [self.evaluate(point) for point in evaluation_set]
        return [metric(evaluation_set, evaluation_results) for metric in metrics] if evaluation_set.size else []

    def evaluate(self, point):
        # A kd-tree can be used to find the cell faster than O(N)
        #cell = min(self.points, key=lambda p: np.sum(np.square(p - point)))
        cell = self.points[0]
        for p in self.points:
            if sum((p - point) ** 2) < sum((cell - point) ** 2):
                cell = p
        return self.function(cell)

    def plot_points(self, axes):
        marker_size = adaptive_marker_size(self.sample_count())
        axes.scatter(self.points[:self.seeds_count, 0], self.points[:self.seeds_count, 1], s=marker_size, c='Red', alpha=0.9)
        axes.scatter(self.points[self.seeds_count:, 0], self.points[self.seeds_count:, 1], s=marker_size, c='Orange', alpha=0.9)

    def plot_partitioning(self, axes):
        for simplex in self.triangulation:
            for i in range(len(simplex)):
                for j in range(i, len(simplex)):
                    axes.plot([self.points[simplex[i]][0], self.points[simplex[j]][0]],
                              [self.points[simplex[i]][1], self.points[simplex[j]][1]],
                              c='Black', linewidth=0.25, alpha=0.9)


class DelaunayInterpolatorBoundaryIntersect(DelaunayInterpolator):
    base_name = "bounded ANNR"

    def query_new_point(self):
        score_list = []
        for simplex in self.triangulation:
            ccenter = circumcenter(self.points[simplex])
            if self.domain.contains(ccenter):
                score_list.append((self.score(self.points[simplex]), ccenter))
            else:
                score_list.append((self.score(self.points[simplex]), self.domain.intersect(barycenter(self.points[simplex]), ccenter)))

        # Can be more efficiently implemented with a p-queue (heap)
        best_point = max(score_list, key=lambda a : a[0])[1]
        self.add_points([best_point])


### DEFER

import sys

sys.path = ['/media/dantrag/data/work/programming/dani/density/dafaq/'] + sys.path

from defer.helpers import *
from defer.variables import Variable
from defer.tree import find_leaf

import warnings
warnings.filterwarnings('ignore')

infinity_limit = 1000000000000

class PartitioningInterpolator(Interpolator):
    approximator = None
    partitions = None
    base_name = "DEFER"

    def __init__(self, function, domain):
        super().__init__(function, domain)
        self.name = f'defer_{self.function.name}'

    def sample_count(self):
        return len(self.partitions)

    # Since the new points cannot be added one by one, iteration_count can be
    # overshot
    def run(self, iteration_count,
            evaluate=False, evaluation_frequency=None,
            evaluation_set=np.array([]), evaluation_metrics=[]):
        print(f"Running {self.base_name} ({self.name})...", flush=True)

        if not evaluate or not evaluation_frequency:
            evaluation_frequency = iteration_count * 2

        x = Variable(
            lower=self.domain.bbox()[0],
            upper=self.domain.bbox()[1],
            name="x")
        variables = Variables([x])
        
        score_curve = []
        self.approximator = construct(
            fn=self.function,
            variables=variables,
            is_log_fn=False,
            num_fn_calls=iteration_count,
            callback=lambda iteration_count, approximator:
                score_curve.append([approximator.num_partitions,
                                    self.evaluate_set(approximator.tree,
                                                      evaluation_set,
                                                      evaluation_metrics)]),
            callback_freq_fn_calls=evaluation_frequency)
        self.partitions = self.approximator.all_partitions()
        
        return score_curve

    def evaluate_set(self, current_tree, evaluation_set, metrics):
        evaluation_results = [self.evaluate_tree(current_tree, point) for point in evaluation_set]
        return [metric(evaluation_set, evaluation_results) for metric in metrics] if evaluation_set.size else []

    def evaluate_tree(self, tree, point):
        partition = find_leaf(tree, point)
        return partition.f
        
    def evaluate(self, point):
        partition = find_leaf(self.approximator.tree, point)
        return partition.f

    def plot_points(self, axes):
        marker_size = adaptive_marker_size(self.sample_count())
        points = np.asarray([partition.domain.center_vector for partition in self.partitions])
        axes.scatter(points[:, 0], points[:, 1], s=marker_size, c='Orange', alpha=0.9)

    def plot_partitioning(self, axes):
        for partition in self.partitions:
            lo = partition.domain.lower_limit_vector
            hi = partition.domain.upper_limit_vector
            axes.plot([lo[0], hi[0]], [lo[1], lo[1]], c='Black', linewidth=0.25)
            axes.plot([lo[0], lo[0]], [lo[1], hi[1]], c='Black', linewidth=0.25)

