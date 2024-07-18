import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_spline_coefficients(P, tangents):
    n = len(P)
    coefficients = []
    for i in range(n):
        Pk = P[i]
        Pk1 = P[(i+1) % n]
        Pk_t = tangents[i]
        Pk1_t = tangents[(i+1) % n]
        a = 2*Pk - 2*Pk1 + Pk_t + Pk1_t
        b = -3*Pk + 3*Pk1 - 2*Pk_t - Pk1_t
        c = Pk_t
        d = Pk
        coefficients.append((a, b, c, d))
    return coefficients

def evaluate_spline(coefficients, t):
    a, b, c, d = coefficients
    return a*t**3 + b*t**2 + c*t + d

# Example data points in 3D space
P = np.array([
    [0, 0, 0],  # P1
    [1, 2, 1],  # P2
    [2, 3, 0],  # P3
    [3, 5, 2],  # P4
    [4, 2, 1],  # P5
    [5, 0, 0]   # P6
])
n = len(P)

# Set up and solve the system for periodic boundary conditions
A = np.zeros((n, n))
B = np.zeros((n, 3))

# Fill the matrix A and vector B
for i in range(n):
    A[i, i] = 4
    A[i, (i-1) % n] = 1
    A[i, (i+1) % n] = 1
    B[i] = 3 * (P[(i+1) % n] - P[(i-1) % n])

# Solve for the tangent vectors
tangents = np.linalg.solve(A, B)

# Compute the spline coefficients
coefficients = compute_spline_coefficients(P, tangents)

# Plot the spline
t_values = np.linspace(0, 1, 100)
spline_points = []

for i in range(len(coefficients)):
    segment_points = [evaluate_spline(coefficients[i], t) for t in t_values]
    spline_points.extend(segment_points)

spline_points = np.array(spline_points)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(P[:, 0], P[:, 1], P[:, 2], 'o', label='Data Points')
ax.plot(spline_points[:, 0], spline_points[:, 1], spline_points[:, 2], '-', label='Periodic Cubic Spline')
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Periodic Cubic Spline Interpolation in 3D')
plt.show()
