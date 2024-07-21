import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def binomial_coefficient(n, k):
    return math.comb(n, k)

def bernstein_polynomial(i, n, t):
    return binomial_coefficient(n, i) * (t ** i) * ((1 - t) ** (n - i))

def rational_bezier_surface(u, v, control_points, weights):
    n = len(control_points) - 1
    m = len(control_points[0]) - 1
    
    numerator = np.zeros(3)
    denominator = 0.0
    
    for i in range(n + 1):
        for j in range(m + 1):
            B_i_n = bernstein_polynomial(i, n, u)
            B_j_m = bernstein_polynomial(j, m, v)
            weight = weights[i][j]
            point = np.array(control_points[i][j])
            
            numerator += (B_i_n * B_j_m * weight) * point
            denominator += B_i_n * B_j_m * weight
    
    return numerator / denominator

def plot_rational_bezier_surface(control_points, weights, resolution=50):
    u_values = np.linspace(0, 1, resolution)
    v_values = np.linspace(0, 1, resolution)
    
    surface_points = np.zeros((resolution, resolution, 3))
    
    for i, u in enumerate(u_values):
        for j, v in enumerate(v_values):
            surface_points[i, j] = rational_bezier_surface(u, v, control_points, weights)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    X = surface_points[:, :, 0]
    Y = surface_points[:, :, 1]
    Z = surface_points[:, :, 2]
    
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='b', alpha=0.6, edgecolor='none')
    
    # Plot control points
    control_points = np.array(control_points)
    ax.scatter(control_points[:, :, 0], control_points[:, :, 1], control_points[:, :, 2], color='r')
    
    for i in range(control_points.shape[0]):
        for j in range(control_points.shape[1]):
            ax.text(control_points[i, j, 0], control_points[i, j, 1], control_points[i, j, 2],
                    f'({i},{j})', color='k')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Example usage
control_points = [
    [[0, 0, 0], [1, 0, 1], [2, 0, 0]],
    [[0, 1, 1], [1, 1, 2], [2, 1, 1]],
    [[0, 2, 0], [1, 2, 1], [2, 2, 0]]
]

weights = [
    [1, 1, 1],
    [1, 2, 1],
    [1, 1, 1]
]

plot_rational_bezier_surface(control_points, weights)
