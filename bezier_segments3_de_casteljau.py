import numpy as np
import matplotlib.pyplot as plt

def linear_interpolation(p0, p1, t):
    """
    Perform linear interpolation between points p0 and p1 at parameter t.
    """
    return (1 - t) * p0 + t * p1

def bezier_curve_de_casteljau(t, control_points):
    """
    Evaluate the Bézier curve at parameter t using De Casteljau's algorithm.

    Parameters:
    t (float): Parameter in the range [0, 1].
    control_points (list of numpy arrays): Control points.

    Returns:
    numpy array: Point on the Bézier curve at parameter t.
    """
    points = np.array(control_points)
    n = len(points) - 1

    # De Casteljau's algorithm
    for r in range(1, n + 1):
        points = np.array([linear_interpolation(points[i], points[i + 1], t) for i in range(n - r + 1)])
    
    return points[0]

# Define control points
control_points = [np.array([0, 0]), np.array([1, 2]), np.array([3, 3]), np.array([4, 2]), np.array([5, 0]), np.array([6, 1]),np.array([7, 4]),]

# Generate points on the Bézier curve
t_values = np.linspace(0, 1, 100)
curve_points = np.array([bezier_curve_de_casteljau(t, control_points) for t in t_values])

# Plot the control points
control_points = np.array(control_points)
plt.plot(control_points[:, 0], control_points[:, 1], 'ro--', label='Control Points')

# Plot the Bézier curve
plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label='Bézier Curve')

# Add labels and legend
plt.title('Bézier Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# Show plot
plt.show()
