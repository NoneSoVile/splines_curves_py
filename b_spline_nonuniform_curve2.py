import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# Define control points
control_points = np.array([
    [0.0, 0.0],
    [1.0, 2.0],
    [3.0, 3.0],
    [4.0, 0.0],
    [5.0, 2.0]
])

# Degree of the B-spline curve
degree = 3

# Define the non-uniform knot vector
knot_vector = [0, 0, 0, 0,  0.1, 1, 1, 1, 1]

# Create the B-spline object
b_spline = BSpline(knot_vector, control_points, degree)

# Generate points on the B-spline curve
t = np.linspace(0, 1, 100)
curve_points = b_spline(t)

# Plot the B-spline curve and control points
plt.figure()
plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label='B-Spline curve')
plt.plot(control_points[:, 0], control_points[:, 1], 'ro-', label='Control points')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Non-Uniform B-Spline Curve')
plt.grid(True)
plt.show()
