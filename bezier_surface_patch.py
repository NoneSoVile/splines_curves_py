import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def bernstein_poly(i, n, t):
    """ Compute the Bernstein polynomial of degree n at t. """
    return np.math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

def bezier_surface_patch(control_points, u, v):
    """ Compute the Bézier surface point for parameters u and v. """
    m, n = len(control_points) - 1, len(control_points[0]) - 1
    surface_point = np.array([0.0, 0.0, 0.0])
    
    for i in range(m + 1):
        for j in range(n + 1):
            B_i = bernstein_poly(i, m, u)
            B_j = bernstein_poly(j, n, v)
            surface_point += B_i * B_j * control_points[i][j]
    
    return surface_point

# Define control points for a simple Bézier surface (4x4 grid)
control_points = np.array([
    [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
    [[0, 1, 0], [1, 1, 1], [2, 1, 1], [3, 1, 0]],
    [[0, 2, 0], [1, 2, 1], [2, 2, 1], [3, 2, 0]],
    [[0, 3, 0], [1, 3, 0], [2, 3, 0], [3, 3, 0]]
])

# Generate surface points
u_vals = np.linspace(0, 1, 30)
v_vals = np.linspace(0, 1, 30)
u, v = np.meshgrid(u_vals, v_vals)
surface_points = np.zeros((u.shape[0], u.shape[1], 3))

for i in range(u.shape[0]):
    for j in range(u.shape[1]):
        surface_points[i, j] = bezier_surface_patch(control_points, u[i, j], v[i, j])

# Plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(surface_points[:, :, 0], surface_points[:, :, 1], surface_points[:, :, 2], cmap='viridis')
plt.show()
