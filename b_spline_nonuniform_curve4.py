import numpy as np
import matplotlib.pyplot as plt

def b_spline_basis(i, k, t, knots):
    if k == 0:
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
    else:
        denom1 = knots[i + k] - knots[i]
        denom2 = knots[i + k + 1] - knots[i + 1]
        term1 = (t - knots[i]) / denom1 * b_spline_basis(i, k - 1, t, knots) if denom1 != 0 else 0
        term2 = (knots[i + k + 1] - t) / denom2 * b_spline_basis(i + 1, k - 1, t, knots) if denom2 != 0 else 0
        return term1 + term2

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
knot_vector = [0, 0, 0, 0, 0.5,  1, 1, 1, 1]

# Generate points on the B-spline curve
t_values = np.linspace(0, 1, 100)  # t should be within the range of the knot vector without reaching the last knot
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
