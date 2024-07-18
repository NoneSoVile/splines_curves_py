import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def evaluate_ferguson_surface(u, v, control_points, tangent_vectors_u, tangent_vectors_v):
    P = np.zeros(3)  # Initialize position vector
    dPdu = np.zeros(3)  # Initialize partial derivative with respect to u
    dPdv = np.zeros(3)  # Initialize partial derivative with respect to v
    
    # Compute contributions from Hermite basis functions
    for i in range(4):
        for j in range(4):
            Hu = hermite_basis(u, i)  # Hermite basis function for u
            Hv = hermite_basis(v, j)  # Hermite basis function for v
            
            P += Hu * Hv * control_points[i, j]
            dPdu += Hu * Hv * tangent_vectors_u[i, j]
            dPdv += Hu * Hv * tangent_vectors_v[i, j]
    
    return P

def hermite_basis(t, i):
    if i == 0:
        return (1 - 3 * t**2 + 2 * t**3)
    elif i == 1:
        return t * (t - 1)**2
    elif i == 2:
        return t**2 * (3 - 2 * t)
    elif i == 3:
        return t**2 * (t - 1)

# Example control points and tangent vectors
control_points = np.array([
    [[0, 0, 0], [2, 0, 0], [4, 0, 0], [6, 0, 0]],
    [[0, 2, 0], [2, 2, 4], [4, 2, 4], [6, 2, 0]],
    [[0, 4, 0], [2, 4, 4], [4, 4, 4], [6, 4, 0]],
    [[0, 6, 0], [2, 6, 0], [4, 6, 0], [6, 6, 0]]
])

tangent_vectors_u = np.array([
    [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
    [[1, 0, 4], [1, 0, 4], [1, 0, 4], [1, 0, 4]],
    [[1, 0, 4], [1, 0, 4], [1, 0, 4], [1, 0, 4]],
    [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]
])

tangent_vectors_v = np.array([
    [[0, 1, 0], [0, 1, 4], [0, 1, 4], [0, 1, 0]],
    [[0, 1, 0], [0, 1, 4], [0, 1, 4], [0, 1, 0]],
    [[0, 1, 0], [0, 1, 4], [0, 1, 4], [0, 1, 0]],
    [[0, 1, 0], [0, 1, 4], [0, 1, 4], [0, 1, 0]]
])

# Evaluate points on the surface
u_vals = np.linspace(0, 1, 30)
v_vals = np.linspace(0, 1, 30)
surface_points = np.zeros((len(u_vals), len(v_vals), 3))

for i, u in enumerate(u_vals):
    for j, v in enumerate(v_vals):
        surface_points[i, j] = evaluate_ferguson_surface(u, v, control_points, tangent_vectors_u, tangent_vectors_v)

# Plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

U, V = np.meshgrid(u_vals, v_vals)
ax.plot_surface(U, V, surface_points[:,:,0], rstride=1, cstride=1, color='b', alpha=0.5)
ax.plot_surface(U, V, surface_points[:,:,1], rstride=1, cstride=1, color='g', alpha=0.5)
ax.plot_surface(U, V, surface_points[:,:,2], rstride=1, cstride=1, color='r', alpha=0.5)
#ax.plot_surface(U, V, surface_points[:,:,3], rstride=1, cstride=1, color='r', alpha=0.5)

ax.set_xlabel('U')
ax.set_ylabel('V')
ax.set_zlabel('Surface')

plt.show()
