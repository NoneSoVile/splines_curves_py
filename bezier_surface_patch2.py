import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def bernstein_polynomial(i, n, t):
    return np.math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

def bezier_surface_patch(Q, m, n, u_values, w_values):
    U = np.array([[bernstein_polynomial(i, m, u) for i in range(m + 1)] for u in u_values])
    V = np.array([[bernstein_polynomial(j, n, w) for j in range(n + 1)] for w in w_values])

    # Flatten Q to match the dimensions required by np.linalg.solve
    Q_flattened = Q.reshape(-1, Q.shape[-1])
    
    # Solve for control points P
    # Ensure we have a square system
    if np.kron(V, U).shape[0] == Q_flattened.shape[0]:
        P_flattened = np.linalg.solve(np.kron(V, U), Q_flattened)
    else:
        raise ValueError("The number of equations does not match the number of unknowns.")

    P = P_flattened.reshape((m + 1, n + 1, -1))
    return P

def plot_bezier_surface(P, m, n):
    u = np.linspace(0, 1, 100)
    w = np.linspace(0, 1, 100)
    U, W = np.meshgrid(u, w)
    B_u = np.array([[bernstein_polynomial(i, m, u_val) for i in range(m + 1)] for u_val in u])
    B_w = np.array([[bernstein_polynomial(j, n, w_val) for j in range(n + 1)] for w_val in w])
    
    surface = np.zeros((100, 100, P.shape[2]))
    for i in range(m + 1):
        for j in range(n + 1):
            surface += np.outer(B_u[:, i], B_w[:, j]).reshape(100, 100, 1) * P[i, j]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(surface[:, :, 0], surface[:, :, 1], surface[:, :, 2], cmap='viridis')
    plt.show()

# Example data points Q for m = 3, n = 2 (4x3 points)
Q = np.array([
    [[0, 0, 0], [1, 0, 1], [2, 0, 0]],
    [[0, 1, 1], [1, 1, 2], [2, 1, 1]],
    [[0, 2, 0], [1, 2, 1], [2, 2, 0]],
    [[0, 3, 1], [1, 3, 2], [2, 3, 1]]
])

m, n = 3, 2
u_values = [0, 0.25, 0.5, 0.75, 1]
w_values = [0, 0.5, 1]

P = bezier_surface_patch(Q, m, n, u_values, w_values)

print("Control Points P:")
print(P)

plot_bezier_surface(P, m, n)
