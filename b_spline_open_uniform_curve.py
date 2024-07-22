import numpy as np
import matplotlib.pyplot as plt

def cox_de_boor(u, k, i, t):
    """ Cox-de Boor recursion formula for B-spline basis functions """
    if k == 0:
        return 1.0 if t[i] <= u < t[i+1] else 0.0
    else:
        coef1 = (u - t[i]) / (t[i + k] - t[i]) if t[i + k] != t[i] else 0.0
        coef2 = (t[i + k + 1] - u) / (t[i + k + 1] - t[i + 1]) if t[i + k + 1] != t[i + 1] else 0.0
        return coef1 * cox_de_boor(u, k - 1, i, t) + coef2 * cox_de_boor(u, k - 1, i + 1, t)

def b_spline(u, k, P, T):
    """ Construct B-spline curve at parameter value u """
    n = len(P) - 1
    result = np.zeros(2)
    for i in range(n + 1):
        result += cox_de_boor(u, k, i, T) * P[i]
    return result

# Control points
P = np.array([
    [0.0, 0.0],
    [1.0, 2.0],
    [3.0, 3.0],
    [4.0, 0.0],
    [5.0, -1.0],
    [6.0, 2.0]
])

# Degree of the spline
k = 3

# Knot vector
n = len(P) - 1
T = np.concatenate(([0] * (k + 1), np.arange(1, n - k + 1), [n - k + 1] * (k + 1)))

# Generate B-spline curve
u_vals = np.linspace(T[k], T[n+1], 100)
curve = np.array([b_spline(u, k, P, T) for u in u_vals])

# Plot control points and B-spline curve
plt.plot(P[:, 0], P[:, 1], 'ro-', label='Control Points')
plt.plot(curve[:, 0], curve[:, 1], 'b-', label='B-spline Curve')
plt.legend()
plt.title("Open Uniform B-Spline Curve")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
