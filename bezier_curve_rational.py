import numpy as np
import matplotlib.pyplot as plt

def rational_bezier_curve(t, control_points, weights):
    """
    Evaluate the rational Bézier curve at parameter t using control points and weights.

    Parameters:
    t (float): Parameter in the range [0, 1].
    control_points (list of numpy arrays): Control points.
    weights (list of floats): Weights associated with the control points.

    Returns:
    numpy array: Point on the rational Bézier curve at parameter t.
    """
    points = np.array(control_points)
    w = np.array(weights)
    n = len(points) - 1

    # Compute numerator and denominator
    numerator = sum(
        (np.math.comb(n, i) * (1 - t)**(n - i) * t**i * w[i] * points[i])
        for i in range(n + 1)
    )
    denominator = sum(
        (np.math.comb(n, i) * (1 - t)**(n - i) * t**i * w[i])
        for i in range(n + 1)
    )

    return numerator / denominator

# Define control points and weights
control_points = [np.array([0, 0]), np.array([1, 2]), np.array([3, 3]), np.array([4, 2]), np.array([5, 0])]
weights = [1, 1, 1, 1, 1]  # Equal weights (default)

# Generate points on the rational Bézier curve
t_values = np.linspace(0, 1, 100)
curve_points = np.array([rational_bezier_curve(t, control_points, weights) for t in t_values])

# Plot the control points
control_points_array = np.array(control_points)
plt.plot(control_points_array[:, 0], control_points_array[:, 1], 'ro--', label='Control Points')

# Plot the rational Bézier curve
plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label='Rational Bézier Curve')

# Show the plot
plt.title('Rational Bézier Curve with Equal Weights')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
#plt.show()

# Now vary the weights and plot the changes
weights_list = [
    [1, 1, 1, 1, 1],            # Equal weights
    [1, 2, 13, 2, 1],            # Different weights
    [1, 0.5, 2, 0.5, 1],        # Different weights (emphasizing middle points)
    [2, 1, 1, 1, 2],            # Weights skewed towards end points
]

plt.figure(figsize=(12, 8))

for weights in weights_list:
    curve_points = np.array([rational_bezier_curve(t, control_points, weights) for t in t_values])
    plt.plot(curve_points[:, 0], curve_points[:, 1], label=f'Weights: {weights}')

plt.plot(control_points_array[:, 0], control_points_array[:, 1], 'ro--', label='Control Points')
plt.title('Rational Bézier Curve with Different Weights')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
