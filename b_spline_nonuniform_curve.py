import numpy as np
import matplotlib.pyplot as plt

# Define the knot vector, degree, and control points
knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 4]
degree = 2
control_points = np.array([[0, 0], [1, 2], [3, 3], [4, 0], [5, -1]])

# Cox-de Boor recursion formula
def basis_function(i, p, t, knot_vector):
    if p == 0:
        return 1.0 if knot_vector[i] <= t < knot_vector[i+1] else 0.0
    else:
        coeff1 = (t - knot_vector[i]) / (knot_vector[i+p] - knot_vector[i]) if knot_vector[i+p] != knot_vector[i] else 0
        coeff2 = (knot_vector[i+p+1] - t) / (knot_vector[i+p+1] - knot_vector[i+1]) if knot_vector[i+p+1] != knot_vector[i+1] else 0
        return coeff1 * basis_function(i, p-1, t, knot_vector) + coeff2 * basis_function(i+1, p-1, t, knot_vector)

# Evaluate the B-Spline curve at a given parameter value t
def evaluate_b_spline(t, degree, knot_vector, control_points):
    n = len(control_points) - 1
    p = degree
    point = np.zeros(2)
    for i in range(n+1):
        b = basis_function(i, p, t, knot_vector)
        point += b * control_points[i]
    return point

# Generate points on the B-Spline curve
num_points = 100
t_values = np.linspace(knot_vector[degree], knot_vector[-degree-1], num_points)
curve_points = np.array([evaluate_b_spline(t, degree, knot_vector, control_points) for t in t_values])

# Plot the B-Spline curve and control points
plt.plot(curve_points[:, 0], curve_points[:, 1], label='B-Spline Curve')
plt.plot(control_points[:, 0], control_points[:, 1], 'ro--', label='Control Points')
plt.legend()
plt.show()
