import numpy as np
import matplotlib.pyplot as plt

def catmull_rom_spline(P0, P1, P2, P3, num_points=100):
    """
    Compute Catmull-Rom spline interpolation between P1 and P2.
    P0, P1, P2, P3 are control points.
    num_points is the number of interpolated points to generate.
    """
    # Convert points to numpy arrays
    P0, P1, P2, P3 = map(np.array, [P0, P1, P2, P3])
    
    # Calculate t values
    t = np.linspace(0, 1, num_points)
    
    # Catmull-Rom matrix
    M = 0.5 * np.array([
        [ 0,  2,  0,  0],
        [-1,  0,  1,  0],
        [ 2, -5,  4, -1],
        [-1,  3, -3,  1]
    ])
    
    # Calculate points
    points = np.zeros((num_points, 2))
    for i in range(num_points):
        t_vec = np.array([1, t[i], t[i]**2, t[i]**3])
        points[i] = t_vec.dot(M).dot([P0, P1, P2, P3])
    
    return points

def create_spline(control_points, num_points_per_segment=100):
    """
    Create a Catmull-Rom spline from a list of control points.
    """
    num_segments = len(control_points) - 3
    spline_points = []
    
    for i in range(num_segments):
        P0, P1, P2, P3 = control_points[i:i+4]
        segment_points = catmull_rom_spline(P0, P1, P2, P3, num_points_per_segment)
        spline_points.extend(segment_points)
    
    return np.array(spline_points)

# Example usage
control_points = [
    (0, 0), (1, 5), (4, 4), (7, 6), (9, 1), (12, 3)
]

# Create the spline
spline_points = create_spline(control_points)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(*zip(*control_points), 'ro-', label='Control Points')
plt.plot(spline_points[:, 0], spline_points[:, 1], 'b-', label='Catmull-Rom Spline')
plt.legend()
plt.title('Catmull-Rom Spline')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()