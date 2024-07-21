import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def bernstein_matrix(N):
    return np.array([
        [-1,  3, -3, 1],
        [ 3, -6,  3, 0],
        [-3,  3,  0, 0],
        [ 1,  0,  0, 0]
    ])

def bezier_surface_patch(P, u_values, w_values):
    N = bernstein_matrix(3)
    surface = np.zeros((len(u_values), len(w_values), P.shape[2]))
    
    for i, u in enumerate(u_values):
        U = np.array([u**3, u**2, u, 1]).reshape(1, 4)
        for j, w in enumerate(w_values):
            W = np.array([w**3, w**2, w, 1]).reshape(4, 1)
            surface[i, j] = U @ N @ P @ N.T @ W
            
    return surface

def plot_bezier_surface(surface):
    u = np.linspace(0, 1, surface.shape[0])
    w = np.linspace(0, 1, surface.shape[1])
    U, W = np.meshgrid(u, w)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(U, W, surface[:, :, 2], cmap='viridis')
    plt.show()

# Example control points P (4x4 grid for bicubic Bézier patch)
P = np.array([
    [[0, 0, 0], [1, 0, 1], [2, 0, 0], [3, 0, 1]],
    [[0, 1, 1], [1, 1, 2], [2, 1, 1], [3, 1, 2]],
    [[0, 2, 0], [1, 2, 1], [2, 2, 0], [3, 2, 1]],
    [[0, 3, 1], [1, 3, 2], [2, 3, 1], [3, 3, 2]]
])

u_values = np.linspace(0, 1, 100)
w_values = np.linspace(0, 1, 100)

surface = bezier_surface_patch(P, u_values, w_values)

print("Generated Bézier surface patch:")
print(surface)

plot_bezier_surface(surface)
