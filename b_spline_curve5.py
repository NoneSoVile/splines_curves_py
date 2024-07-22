import numpy as np
import matplotlib.pyplot as plt

def clamp(value, low, high):
    """Clamp value between low and high."""
    return max(low, min(high, value))

def b_spline_basis_function(degree, knot, knots, t):
    """Calculate the value of a single B-spline basis function."""
    if degree == 0:
        # The degree 0 basis function is a piecewise constant function.
        return 1.0 if knot <= t < knots[knot+1] else 0.0
    
    if knots[knot+degree] == knots[knot]:
        c1 = 0.0
    else:
        c1 = (t - knots[knot]) / (knots[knot+degree] - knots[knot]) * b_spline_basis_function(degree-1, knot, knots, t)
    
    if knots[knot+degree+1] == knots[knot+1]:
        c2 = 0.0
    else:
        c2 = (knots[knot+degree+1] - t) / (knots[knot+degree+1] - knots[knot+1]) * b_spline_basis_function(degree-1, knot+1, knots, t)
    
    return c1 + c2

def calculate_b_spline(control_points, degree, knot_vector, num_points):
    """Calculate the B-spline curve."""
    t_values = np.linspace(knot_vector[degree], knot_vector[-degree-1], num_points)
    curve_points = []
    
    for t in t_values:
        x, y = 0.0, 0.0
        for i, (x_i, y_i) in enumerate(control_points):
            basis_value = b_spline_basis_function(degree, i, knot_vector, t)
            x += x_i * basis_value
            y += y_i * basis_value
        curve_points.append((x, y))
    
    curve = np.array(curve_points)
    print(f"Curve shape: {curve.shape}")  # Debugging statement
    
    return curve

# Define the control points
control_points = np.array([[0, 0], [1, 1], [2, -1], [3, 0]])

# Define the degree of the B-spline
degree = 3

# Define the knot vector
knot_vector = [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]

# Number of points to evaluate the spline at
num_points = 100

# Generate the B-spline curve
curve = calculate_b_spline(control_points, degree, knot_vector, num_points)

# Plot the control points and the B-spline curve using matplotlib
plt.figure()
plt.plot(control_points[:, 0], control_points[:, 1], 'ro', label='Control Points')
plt.plot(curve[:, 0], curve[:, 1], 'b-', label='B-Spline Curve')

plt.legend(loc='best')
plt.title('Cubic B-Spline Curve')
plt.grid(True)
plt.axis('equal')
plt.show()