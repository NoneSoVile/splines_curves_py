import numpy as np
import matplotlib.pyplot as plt

def compute_spline_coefficients(P, tangents):
    n = len(P)
    coefficients = []
    for i in range(n-1):
        Pk = P[i]
        Pk1 = P[i+1]
        Pk_t = tangents[i]
        Pk1_t = tangents[i+1]
        a = 2*Pk - 2*Pk1 + Pk_t + Pk1_t
        b = -3*Pk + 3*Pk1 - 2*Pk_t - Pk1_t
        c = Pk_t
        d = Pk
        coefficients.append((a, b, c, d))
    return coefficients

def evaluate_spline(coefficients, t):
    a, b, c, d = coefficients
    return a*t**3 + b*t**2 + c*t + d

# Original data points
P = np.array([
    [0., 0],  # P1
    [1, 0],  # P2
    [1, 1],  # P3
    [0, 1]   # P4
])

# New data points to add
new_points = np.array([
    [0.5, -1],  # P5
    [0.7, -2]   # P6
])

# Combine original and new data points
#P = np.vstack([P, new_points])
# Positions where new points will be inserted
positions = [1, 2]

# Inserting new points at the specified positions
for new_point, position in zip(new_points, positions):
    P = np.insert(P, position, new_point, axis=0)
    
    
# User-provided tangent vectors at the endpoints
P1_t = np.array([1, -1])
Pn_t = np.array([-1, 0])

# Set up and solve the system for the new tangents
n = len(P) - 2  # Number of interior tangents
A = np.zeros((n, n))
B = np.zeros((n, 2))

for i in range(n):
    if i > 0:
        A[i, i-1] = 1
    A[i, i] = 4
    if i < n - 1:
        A[i, i+1] = 1
    B[i] = 3 * (P[i+2] - P[i])

B[0] -= P1_t
B[-1] -= Pn_t
print(f"data P:\n{P}")
print(f"matrix A:\n{A}")
print(f"matrix B:\n{B}")
interior_tangents = np.linalg.solve(A, B)
tangents = np.vstack([P1_t, interior_tangents, Pn_t])
coefficients = compute_spline_coefficients(P, tangents)

# Plot the spline
t_values = np.linspace(0, 1, 100)
spline_points = []

for i in range(len(coefficients)):
    segment_points = [evaluate_spline(coefficients[i], t) for t in t_values]
    spline_points.extend(segment_points)

spline_points = np.array(spline_points)

plt.plot(P[:, 0], P[:, 1], 'o', label='Data Points')
plt.plot(spline_points[:, 0], spline_points[:, 1], '-', label='Spline')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Hermite Spline Interpolation with Additional Points')
plt.show()
