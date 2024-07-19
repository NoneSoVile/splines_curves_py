import numpy as np
import matplotlib.pyplot as plt

def parabola_through_3_points(P0, P1, P2):
    """
    Calculate coefficients of a parabola y = ax^2 + bx + c
    passing through 3 points.
    """
    x0, y0 = P0
    x1, y1 = P1
    x2, y2 = P2
    
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2*x2 * (y0 - y1) + x1*x1 * (y2 - y0) + x0*x0 * (y1 - y2)) / denom
    c = (x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1 + x0 * x1 * (x0 - x1) * y2) / denom
    
    return a, b, c

def catmull_rom_segment(P0, P1, P2, P3, num_points=100):
    """
    Compute Catmull-Rom spline segment between P1 and P2 using blended parabolas.
    """
    # Calculate coefficients for the two parabolas
    a1, b1, c1 = parabola_through_3_points(P0, P1, P2)
    a2, b2, c2 = parabola_through_3_points(P1, P2, P3)
    
    # Parametric values
    t = np.linspace(0, 1, num_points)
    
    # Blend the two parabolas
    x = (1 - t) * (a1 * t**2 + b1 * t + c1) + t * (a2 * t**2 + b2 * t + c2)
    y = (1 - t) * P1[1] + t * P2[1]
    
    return np.column_stack((x, y))

def create_catmull_rom_spline(control_points, num_points_per_segment=100):
    """
    Create a Catmull-Rom spline from a list of control points using blended parabolas.
    """
    num_segments = len(control_points) - 3
    spline_points = []
    
    for i in range(num_segments):
        P0, P1, P2, P3 = control_points[i:i+4]
        segment_points = catmull_rom_segment(P0, P1, P2, P3, num_points_per_segment)
        spline_points.extend(segment_points)
    
    return np.array(spline_points)

# Example usage
control_points = [
    (0, 0), (1, 2), (3, 3), (5, 6), (6, 3), (7.1, 2)
]

# Create the spline
spline_points = create_catmull_rom_spline(control_points)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(*zip(*control_points), 'ro-', label='Control Points')
plt.plot(spline_points[:, 0], spline_points[:, 1], 'b-', label='Catmull-Rom Spline')
plt.legend()
plt.title('Catmull-Rom Spline (Blended Parabolas Method)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()