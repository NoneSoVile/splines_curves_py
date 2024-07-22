import numpy as np
import matplotlib.pyplot as plt

def basis_function(i, p, u, U):
    if p == 0:
        return 1.0 if U[i] <= u < U[i+1] else 0.0
    else:
        left = 0.0 if U[i+p] == U[i] else (u - U[i]) / (U[i+p] - U[i]) * basis_function(i, p-1, u, U)
        right = 0.0 if U[i+p+1] == U[i+1] else (U[i+p+1] - u) / (U[i+p+1] - U[i+1]) * basis_function(i+1, p-1, u, U)
        return left + right

def b_spline_curve(P, p, U, u):
    n = len(P) - 1
    C = np.zeros(2)
    for i in range(n+1):
        C += basis_function(i, p, u, U) * P[i]
    return C

# Control points
P = np.array([
    [0.0, 0.0],
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 2.0],
    [4.0, 0.0]
])

# Degree
p = 4

# Knot vector
U = [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]

# Plotting the B-spline curve
u_vals = np.linspace(U[p], U[len(U)-p-1], 100)
curve = np.array([b_spline_curve(P, p, U, u) for u in u_vals])

plt.plot(curve[:, 0], curve[:, 1], label='B-spline Curve')
plt.plot(P[:, 0], P[:, 1], 'o--', label='Control Points')
plt.legend()
plt.show()
