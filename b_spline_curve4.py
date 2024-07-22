import numpy as np
import matplotlib.pyplot as plt

def bspline_basis(i, k, t, knots):
    """
    Compute the B-spline basis function value.
    
    i: index of the control point
    k: degree of the curve
    t: parameter value
    knots: knot vector
    """
    if k == 0:
        return 1.0 if knots[i] <= t <= knots[i+1] else 0.0
    
    if knots[i+k] == knots[i]:
        c1 = 0.0
    else:
        c1 = (t - knots[i]) / (knots[i+k] - knots[i]) * bspline_basis(i, k-1, t, knots)
    
    if knots[i+k+1] == knots[i+1]:
        c2 = 0.0
    else:
        c2 = (knots[i+k+1] - t) / (knots[i+k+1] - knots[i+1]) * bspline_basis(i+1, k-1, t, knots)
    
    return c1 + c2

def bspline_curve(control_points, degree, num_points=100):
    """
    Compute points on a B-spline curve.
    
    control_points: list of control points
    degree: degree of the curve
    num_points: number of points to evaluate on the curve
    """
    n = len(control_points) - 1
    knots = np.concatenate([np.zeros(degree),
                            np.linspace(0, 1, n - degree + 2),
                            np.ones(degree)])
    
    curve_points = []
    for t in np.linspace(0, 1, num_points):
        point = np.zeros(2)
        for i in range(n + 1):
            basis = bspline_basis(i, degree, t, knots)
            point += basis * np.array(control_points[i])
        curve_points.append(point)
    
    return np.array(curve_points)

# Example usage
control_points = [(0, 0), (1, 4), (2, 0), (3, 4), (4, 0)]
degree = 3

curve_points = bspline_curve(control_points, degree)
print(curve_points)
# Plotting
plt.figure(figsize=(10, 6))
control_points = np.array(control_points)
plt.plot(control_points[:, 0], control_points[:, 1], 'ro-', label='Control Points')
plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label='B-spline Curve')
plt.legend()
plt.title(f'B-spline Curve (Degree {degree})')
plt.grid(True)
plt.axis('equal')
plt.show()