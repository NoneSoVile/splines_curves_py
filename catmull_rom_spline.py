import numpy as np
import matplotlib.pyplot as plt

def catmull_rom_spline(P0, P1, P2, P3, n_points=100):
    """
    Returns points evaluated along the Catmull-Rom spline defined by four points.
    
    Parameters:
        P0, P1, P2, P3: coordinates of the 4 points
        n_points: number of points to return
    
    Returns:
        A list of n_points, evaluated along the spline.
    """
    # Convert the points to numpy array for vectorized operations
    P0, P1, P2, P3 = map(np.array, [P0, P1, P2, P3])
    
    # Generate the parameter t for the spline segment
    t = np.linspace(0, 1, n_points)
    
    # Calculate the x and y components separately
    x = 0.5 * ((-t**3 + 2*t**2 - t) * P0[0] + 
               (3*t**3 - 5*t**2 + 2) * P1[0] +
               (-3*t**3 + 4*t**2 + t) * P2[0] +
               (t**3 - t**2) * P3[0])
    
    y = 0.5 * ((-t**3 + 2*t**2 - t) * P0[1] + 
               (3*t**3 - 5*t**2 + 2) * P1[1] +
               (-3*t**3 + 4*t**2 + t) * P2[1] +
               (t**3 - t**2) * P3[1])
    
    return np.column_stack((x, y))

def generate_catmull_rom(points, n_points_per_segment=100):
    """
    Generates a Catmull-Rom spline from a list of points.
    
    Parameters:
        points: a list of (x,y) tuples representing the control points
        n_points_per_segment: number of points to interpolate per segment
    
    Returns:
        A list of interpolated points along the spline.
    """
    # Generate the spline segments
    spline_points = []
    for i in range(len(points) - 3):
        p0, p1, p2, p3 = points[i:i+4]
        spline_points.append(catmull_rom_spline(p0, p1, p2, p3, n_points=n_points_per_segment))
    
    # Concatenate the segments
    return np.vstack(spline_points)

# Example usage
control_points = [(1, 0), (3, 1), (6, 2), (7, 3),(8, 3),(8, 5),(2, 3), (1, 4)]
spline_points = generate_catmull_rom(control_points, n_points_per_segment=100)

# Plotting the control points
plt.figure(figsize=(10, 6))
plt.plot(*zip(*control_points), marker='o', color='red', linestyle='none')

# Plotting the spline
plt.plot(spline_points[:, 0], spline_points[:, 1], color='blue')

# Additional plotting options can be added here, such as setting labels and titles
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Catmull-Rom Spline')
plt.grid(True)

# Show the plot
plt.show()
