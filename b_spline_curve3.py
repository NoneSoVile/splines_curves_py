import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

def bspline(control_points, degree, num_points):
    # Calculate the knot vector
    num_knots = len(control_points) + degree + 1
    knot_vector = np.linspace(0, 1, num_knots)

    # Create the B-spline object
    spline = BSpline(knot_vector, control_points, degree)

    # Evaluate the spline at num_points
    u = np.linspace(0, 1, num_points)
    curve = spline(u)

    return curve

# Define the control points
control_points = np.array([[0, 0], [1, 1], [2, -1], [3, 0]])

# Set the degree of the B-spline
degree = 3

# Number of points to evaluate the spline at
num_points = 100

# Generate the B-spline curve
curve = bspline(control_points, degree, num_points)
print(curve)
# Plot the control points and the B-spline curve
plt.figure()
plt.plot(control_points[:, 0], control_points[:, 1], 'o', label='Control Points')
plt.plot(curve[:, 0], curve[:, 1], label='B-Spline Curve')
plt.legend(loc='best')
plt.title('Cubic B-Spline Curve')
plt.grid(True)
plt.axis('equal')
plt.show()