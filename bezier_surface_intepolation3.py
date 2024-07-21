import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import comb

def bernstein_poly(i, n, t):
    """ Compute the Bernstein polynomial of degree n at t. """
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

def bernstein_basis(u, v, m, n):
    """ Calculate Bernstein basis functions for a given (u, v) """
    basis = []
    for i in range(m + 1):
        for j in range(n + 1):
            basis.append(bernstein_poly(i, m, u) * bernstein_poly(j, n, v))
    return basis

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

def plot_surface(control_points, interpolation_points, title):
    """ Plot the Bézier surface given the control points. """
    u = np.linspace(0, 1, 100)
    v = np.linspace(0, 1, 100)
    U, V = np.meshgrid(u, v)
    Z = np.zeros_like(U)
    
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            Z[i, j] = bezier_surface_patch(control_points, U[i, j], V[i, j])[2]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(U, V, Z, cmap='viridis', alpha=0.6)
    
    # Plot control points
    control_points_array = np.array(control_points)
    ax.scatter(control_points_array[:, 0], control_points_array[:, 1], control_points_array[:, 2], color='red', label='Control Points')
    
    # Plot interpolation points
    ax.scatter(interpolation_points[:, 0], interpolation_points[:, 1], interpolation_points[:, 2], color='blue', label='Interpolation Points')
    
    ax.set_title(title)
    ax.legend()
    plt.show()

# New interpolation points (adding more points)
interpolation_points = np.array([
    [0, 0, 0],    # (u=0, v=0)
    [0.5, 0, 0.5],  # (u=0.5, v=0)
    [1, 0, 1],    # (u=1, v=0)
    [0, 0.5, 0.5],  # (u=0, v=0.5)
    [0.5, 0.5, 1],  # (u=0.5, v=0.5)
    [1, 0.5, 1.5],  # (u=1, v=0.5)
    [0, 1, 2],    # (u=0, v=1)
    [0.5, 1, 2.5],  # (u=0.5, v=1)
    [1, 1, 3]     # (u=1, v=1)
])

# Number of control points in each direction
m, n = 2, 2  # This implies 3x3 control points (9 points total)

# Set up the matrix A and vector B
num_control_points = (m + 1) * (n + 1)
A = np.zeros((len(interpolation_points) * 3, num_control_points))  # 3 coordinates * number of interpolation points
B = np.zeros((len(interpolation_points) * 3, 3))

# Populate matrix A and vector B using interpolation points
for idx, (u, v) in enumerate([(0,0), (0.5,0), (1,0), (0,0.5), (0.5,0.5), (1,0.5), (0,1), (0.5,1), (1,1)]):
    basis = bernstein_basis(u, v, m, n)
    for k in range(3):  # x, y, z coordinates
        A[idx * 3 + k, :] = basis
        B[idx * 3 + k, :] = interpolation_points[idx]

# Solve for control points
control_points = np.linalg.lstsq(A, B, rcond=None)[0]

# Reshape control points into 3x3 grid for plotting
control_points_grid = control_points.reshape((m + 1, n + 1, 3))

# Plot the surface with the calculated control points
plot_surface(control_points_grid, interpolation_points, 'Bézier Surface with Control and Interpolation Points')
