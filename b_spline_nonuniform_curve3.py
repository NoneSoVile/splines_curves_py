import numpy as np
import matplotlib.pyplot as plt

# Define the Cox-de Boor recursion formula for B-spline basis functions
def b_spline_basis(i, k, t, knots):
    if k == 0:
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
    else:
        c1 = (t - knots[i]) / (knots[i + k] - knots[i]) if knots[i + k] != knots[i] else 0
        c2 = (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]) if knots[i + k  + 1] != knots[i + 1] else 0
        return c1 * b_spline_basis(i, k - 1, t, knots) + c2 * b_spline_basis(i + 1, k - 1, t, knots)

# Define control points
control_points = np.array([
    [0.0, 0.0],
    [1.0, 2.0],
    [3.0, 3.0],
    [4.0, 0.0],
    [5.0, 2.0],
])

# Degree of the B-spline curve
degree = 3

# Define the non-uniform knot vector
knot_vector = [0, 0,  0., 0., 0.5, 1., 1, 1., 1.]

# Generate points on the B-spline curve
t_values = np.linspace(0, 1, 100)
curve_points = np.zeros((len(t_values), 2))

for j, t in enumerate(t_values):
    point = np.zeros(2)
    for i in range(len(control_points)):
        basis = b_spline_basis(i, degree, t, knot_vector)
        point += basis * control_points[i]
    curve_points[j] = point
print(curve_points)
# Plot the B-spline curve and control points
plt.figure()
plt.plot(curve_points[:-1, 0], curve_points[:-1, 1], 'b-', label='B-Spline curve')
plt.plot(control_points[:, 0], control_points[:, 1], 'ro-', label='Control points')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Non-Uniform B-Spline Curve')
plt.grid(True)
plt.show()
