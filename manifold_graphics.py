import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import random
import matplotlib
from matplotlib import cm
from matplotlib.patches import Polygon
from matplotlib import collections  as mc
import pylab as pl
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Polygon
from matplotlib import cm
from matplotlib.colors import rgb_to_hsv
from scipy.spatial import Delaunay
import math




# def adaptive_marker_size(count):
#     return min(20, 200 // math.sqrt(count))

T = 100
pts = np.load('latent2_lambda_1.0_239/afi_data.npy')[:T]
values = np.load('latent2_lambda_1.0_239/afi_values.npy')[:T]
#pts = np.vstack(( np.array([[-1., -1.], [ 1., -1.], [-1.,  1.], [ 1.,  1.]]), pts ))
pts_extended = np.vstack((pts, np.array([[-10., -10.], [ 10., -10.], [-10.,  10.], [ 10.,  10.]]) ))
#values = np.hstack((np.array( [158.72572327, 87.73779297, 146.96255493, 306.80929565] ), values ))

print(values.mean(), values.max())
# print(pts.shape, values.shape)

dela = Delaunay(pts)
simplices = dela.simplices

vor = Voronoi(pts_extended)
vertices = vor.vertices
edge_idxs = vor.ridge_vertices
cells = vor.regions
point_region = vor.point_region


# fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='black',
#                 line_width=1, line_alpha=.8, point_size=0)
#ax = plt.gca()
fig, ax = plt.subplots(1,1)

for idx_cell, cell in enumerate(cells):
    if (-1 not in cell) and (cell != []): #and (idx_cell not in big_cells):
        value = values[point_region.tolist().index(idx_cell)]/750
        ax.add_patch(Polygon([vertices[i].tolist() for i in cell], facecolor=cm.Blues(value), alpha=1., linewidth=.25, edgecolor='Black' ))

# for idx_simp, simp in enumerate(simplices):
#     vertices = pts[simp]
#     ax.add_patch(Polygon(vertices, facecolor='None', alpha=.9, linewidth=.3, edgecolor='Black'))


ax.scatter(pts[:, 0], pts[:, 1], color='Orange', alpha=.9, zorder=2, s=5.) #adaptive_marker_size(T)) #color=cm.gist_yarg(np.array(list(range(len(pts))))/len(pts))
for i in range(len(pts)):
    ax.annotate(str(pts[i]), (pts[i, 0], pts[i, 1]))


plt.xlim(-1,1)
plt.ylim(-1,1)
ax.set_aspect('equal', adjustable='box')
#fig.set_dpi(300)
plt.axis('off')
plt.savefig(f'latent_{T}.png', dpi=500, bbox_inches='tight', pad_inches=0.05)
plt.show()
