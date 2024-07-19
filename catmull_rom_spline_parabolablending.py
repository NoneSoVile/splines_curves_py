import numpy as np
import matplotlib.pyplot as plt

def parabola_points(p0, p1, p2, num_points=100):
    # Compute the parabolic blend between three points
    t = np.linspace(0, 1, num_points)
    a = p0
    b = p1
    c = p2

    # Quadratic coefficients for the parabola
    q1 = (1 - t)**2
    q2 = 2 * (1 - t) * t
    q3 = t**2

    # Parabolic interpolation
    points = q1[:, None] * a + q2[:, None] * b + q3[:, None] * c

    return points

def catmull_rom_spline_parabolas(P, num_points=100):
    all_points = []

    # Loop over each set of 4 consecutive control points
    for i in range(len(P) - 3):
        P0, P1, P2, P3 = P[i:i+4]
        
        # Generate parabolas for each segment
        segment1 = parabola_points(P0, P1, P2, num_points)
        segment2 = parabola_points(P1, P2, P3, num_points)
        
        # Combine segments, blend by averaging
        t_values = np.linspace(0, 1, num_points)
        blend_factor = t_values
        blended_segment = (1 - blend_factor[:, None]) * segment1 + blend_factor[:, None] * segment2
        
        all_points.append(blended_segment)

    # Concatenate all segments
    all_points = np.vstack(all_points)
    
    return all_points

# Example control points (more than four)
control_points = np.array([
    [0, 0],
    [1, 2],
    [3, 3],
    [4, 0],
    [5, -1],
    [6, 2],
    [7, 0]
])

# Generate the Catmull-Rom spline curve using parabolas
spline_points = catmull_rom_spline_parabolas(control_points)

# Plot the result
plt.plot(spline_points[:, 0], spline_points[:, 1], label='Catmull-Rom Spline (Parabolas)')
plt.scatter(control_points[:, 0], control_points[:, 1], color='red', label='Control Points')
plt.legend()
plt.show()
