import numpy as np
import math

def simplex_volume(simplex_points):
    #Cayley-Menger determinant (https://mathworld.wolfram.com/Cayley-MengerDeterminant.html)
    dist_matrix = ((np.expand_dims(simplex_points, axis=0) - np.expand_dims(simplex_points, axis=1))**2).sum(axis=-1)
    B_hat = np.zeros((dist_matrix.shape[0] + 1, dist_matrix.shape[1] + 1))
    B_hat[1:, 1:] = dist_matrix
    B_hat[:, 0] = np.ones(B_hat[:, 0].shape)
    B_hat[0, :] = np.ones(B_hat[0, :].shape)
    B_hat[0,0] = 0
    j = len(simplex_points)-1
    det = np.linalg.det(B_hat)
    squared_score = ((-1)**(j+1)/((2**j)*(math.factorial(j)**2)))*det
    return math.sqrt(squared_score)

def barycenter(simplex_points):
    return simplex_points.mean(axis=0)

def circumcenter(points):
    from numpy import bmat, hstack, dot, ones, zeros, sum, asarray
    from numpy.linalg import solve, norm

    pts = asarray(points)
    rows, cols = pts.shape
    assert(rows <= cols + 1)

    A = bmat([[ 2 * dot(pts, pts.T), ones((rows, 1)) ],
              [ ones((1, rows)),   zeros((1, 1)) ]])

    b = hstack((sum(pts * pts, axis=1), ones((1))))
    x = solve(A, b)
    bary_coords = x[:-1] #barycentric coordinates of circumcenter
    return (pts * np.expand_dims(bary_coords, axis=1)).sum(axis=0)

 
