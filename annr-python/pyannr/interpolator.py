from optparse import Values
from random import seed
from pyannr.function import Function
from pyannr.domain import Domain, RectangularDomain
from pyannr.geometry_utils import *

from abc import ABCMeta, abstractmethod

import math
import numpy as np
from numpy.random import random, choice
from scipy.spatial import Delaunay, Voronoi, ConvexHull

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import cm
from matplotlib.colors import to_rgba

from tqdm import tqdm

from pyannr.utils import load_plot_style, reset_plot_style, adaptive_marker_size


line_width = 0.5
alpha = 0.9

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

    @abstractmethod
    def evaluate_set(self, evaluation_set, metrics):
        raise NotImplementedError

    def run(self, iteration_count, continues=False,
            evaluate=False, evaluation_frequency=1,
            evaluation_set=np.array([]), evaluation_metrics=[]):
        raise NotImplementedError

    def plot_extra(self, axes, **args):
        pass

    def plot_approximate(self, axes, **args):
        pass

    def plot(self, axes, points=True, partitioning=True, **args):
        if points:
            self.plot_points(axes, **args)
        if partitioning:
            self.plot_partitioning(axes, **args)

        self.plot_extra(axes, **args)
        

    def save_plot(self, filename='auto', title='', points=True, partitioning=True, mode='approx', **args):
        load_plot_style()

        figure, axes = plt.subplots()
        self.plot(axes, points=points, partitioning=partitioning, **args)

        if mode == 'approx':
            self.plot_approximate(axes, **args)
        elif mode == 'gt':
            self.function.draw(axes, self.domain)

        if 'show_axes' in args:
            if not args['show_axes']:
                axes.set_axis_off()
        axes.set_title(title)
        axes.set_xlim(self.domain.bbox()[0][0], self.domain.bbox()[1][0])
        axes.set_ylim(self.domain.bbox()[0][1], self.domain.bbox()[1][1])
        axes.set_aspect(1)
        
        figure.tight_layout()

        if filename == 'auto':
            filename = f'{self.name}_{self.sample_count()}.png'
        figure.savefig(filename,
                       dpi=600, bbox_inches='tight', pad_inches=0.05)

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

    def __init__(self, function, domain, seeds=10, add_corners=True, Lambda='auto'):
        super().__init__(function, domain)

        self.points = np.empty([0, domain.dimensionality()])

        if isinstance(seeds, int):
            seeds = self.domain.sample_random(seeds)
            if add_corners:
                seeds = np.concatenate((seeds, domain.corners()), axis=0)

        self.seeds_count = len(seeds)
        if len(seeds):
            self.add_points(seeds)
        
        if Lambda == 'auto':
            seed_values = [function(x) for x in seeds]
            value_scale = max(seed_values) - min(seed_values)
            value_scale = value_scale if value_scale else 1
            Lambda = domain.volume() / value_scale

        self.Lambda = Lambda
        self.name = f'annr_{function.name}_{hex(id(self))}'
        self.iteration_offset = self.seeds_count

    def sample_count(self):
        return len(self.points)

    def add_points(self, new_points, manual_addition=False):
        self.points = np.concatenate((self.points, new_points), axis=0)
        
        # Recomputing entire triangulation. Can be done more efficiently with Delaunay update algos
        self.triangulation = Delaunay(self.points).simplices
        
        if not manual_addition:
            if self.sample_count() % 100 == 0:
                pass
                # uncomment to save intermediate results
                #self.save_points_to_file()
                #self.snapshots.append(self.points.copy())
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

    def run(self, iteration_count, continues=False,
            evaluate=False, evaluation_frequency=None,
            evaluation_set=np.array([]), evaluation_metrics=[]):      
        print(f"Running {self.base_name} ({self.name})...", flush=True)

        if not continues:
            # reset initialization
            self.points = self.points[:self.seeds_count]
            self.iteration_offset = self.seeds_count
            self.triangulation = Delaunay(self.points).simplices

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

    def save_points_to_file(self, filename="auto"):
        if filename == "auto":
            filename = f'{self.name}_{self.sample_count()}.npy'
        np.save(filename, self.points, allow_pickle=True)
    
    def load_points_from_file(self, filename):
        points = np.load(filename, allow_pickle=True)
        self.points = np.empty([0, self.domain.dimensionality()])
        self.add_points(points, manual_addition=True)


    def plot_points(self, axes, **args):
        seed_color = 'Red' if (not 'highlight_seeds' in args) or args['highlight_seeds'] else 'Orange'

        marker_size = adaptive_marker_size(self.sample_count())
        axes.scatter(self.points[:self.seeds_count, 0], self.points[:self.seeds_count, 1], s=marker_size, zorder=10, c=seed_color, alpha=alpha)
        axes.scatter(self.points[self.seeds_count:, 0], self.points[self.seeds_count:, 1], s=marker_size, zorder=10, c='Orange', alpha=alpha)

    def plot_approximate(self, axes, **args):
        # expand the bounding box to avoid having empty areas
        double_bbox_corners = (self.domain.corners() - np.average(self.domain.corners())) * 2 + np.average(self.domain.corners())
        pts = np.vstack((self.points, double_bbox_corners))
        values = [self.function(x) for x in pts]
        max_value = max(values[:-len(double_bbox_corners)])
        max_value *= 1.05

        if max_value == 0:
            max_value = 1

        voronoi = Voronoi(pts)
        vertices = voronoi.vertices
        cells = voronoi.regions
        point_region = voronoi.point_region

        for idx_cell, cell in enumerate(cells):
            if (-1 not in cell) and (cell != []):
                value = values[point_region.tolist().index(idx_cell)]
                axes.add_patch(Polygon([vertices[i].tolist() for i in cell],
                               facecolor=cm.Blues(value / max_value, alpha), linewidth=line_width, edgecolor=(1, 1, 1, 0)))

    def plot_delaunay(self, axes, **args):
        for simplex in self.triangulation:
            for i in range(len(simplex)):
                for j in range(i, len(simplex)):
                    axes.plot([self.points[simplex[i]][0], self.points[simplex[j]][0]],
                              [self.points[simplex[i]][1], self.points[simplex[j]][1]],
                              c='Black', linewidth=line_width)

    def plot_extra(self, axes, **args):
        super().plot_extra(axes, **args)
        if 'delaunay' in args:
            if args['delaunay']:
                self.plot_delaunay(axes)

    def plot_partitioning(self, axes, **args):
        double_bbox_corners = (self.domain.corners() - np.average(self.domain.corners())) * 2 + np.average(self.domain.corners())
        pts = np.vstack((self.points, double_bbox_corners))

        voronoi = Voronoi(pts)
        vertices = voronoi.vertices
        cells = voronoi.regions

        for idx_cell, cell in enumerate(cells):
            if (-1 not in cell) and (cell != []):
                axes.add_patch(Polygon([vertices[i].tolist() for i in cell],
                               facecolor=(1, 1, 1, 0),
                               linewidth=line_width, edgecolor=to_rgba('Black', alpha),
                               zorder=5))

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

from pydefer.helpers import *
from pydefer.variables import Variable
from pydefer.tree import find_leaf

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
    def run(self, iteration_count, continues=False,
            evaluate=False, evaluation_frequency=None,
            evaluation_set=np.array([]), evaluation_metrics=[]):
        print(f"Running {self.base_name} ({self.name})...", flush=True)

        if continues:
            raise NotImplementedError('Default DEFER implementations does not support consecutive runs')

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
                                    self.evaluate_set(evaluation_set,
                                                      evaluation_metrics,
                                                      approximator.tree)]),
            callback_freq_fn_calls=evaluation_frequency)
        self.partitions = self.approximator.all_partitions()
        
        return score_curve

    def evaluate_set(self, evaluation_set, metrics, current_tree=None):
        if not current_tree:
            current_tree = self.approximator.tree
        evaluation_results = [self.evaluate_tree(current_tree, point) for point in evaluation_set]
        return [metric(evaluation_set, evaluation_results) for metric in metrics] if evaluation_set.size else []

    def evaluate_tree(self, tree, point):
        partition = find_leaf(tree, point)
        return partition.f
        
    def evaluate(self, point):
        partition = find_leaf(self.approximator.tree, point)
        return partition.f

    def plot_points(self, axes, **args):
        marker_size = adaptive_marker_size(self.sample_count())
        points = np.asarray([partition.domain.center_vector for partition in self.partitions])
        axes.scatter(points[:, 0], points[:, 1], s=marker_size, c='Orange', zorder=10, alpha=alpha)

    def plot_partitioning(self, axes, **args):
        for partition in self.partitions:
            lo = partition.domain.lower_limit_vector
            up = partition.domain.upper_limit_vector
            rectangle = RectangularDomain(lo, up).corners()
            rectangle = rectangle[ConvexHull(rectangle).vertices]
            axes.add_patch(Polygon(rectangle,
                           facecolor=(1, 1, 1, 0),
                           linewidth=line_width, edgecolor=to_rgba('Black', alpha),
                           zorder=5))

    def plot_approximate(self, axes, **args):
        values = []
        for partition in self.partitions:
            values.append(partition.f)

        max_value = max(values) * 1.05
        if max_value == 0:
            max_value = 1

        for partition in self.partitions:
            lo = partition.domain.lower_limit_vector
            up = partition.domain.upper_limit_vector
            rectangle = RectangularDomain(lo, up).corners()
            rectangle = rectangle[ConvexHull(rectangle).vertices]
            axes.add_patch(Polygon(rectangle,
                           facecolor=cm.Blues(partition.f / max_value, alpha),
                           linewidth=line_width, edgecolor=(1, 1, 1, 0)))
