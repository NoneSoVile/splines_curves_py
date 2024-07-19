import numpy as np
import matplotlib.pyplot as plt

def catmull_rom_spline(P, num_points=100):
    # Prepare to store the results
    all_points = []
    
    # Define the Catmull-Rom matrix
    CR_matrix = 0.5 * np.array([
        [-1,  3, -3,  1],
        [ 2, -5,  4, -1],
        [-1,  0,  1,  0],
        [ 0,  2,  0,  0]
    ])
    
    # Loop over each set of 4 consecutive control points
    for i in range(len(P) - 3):
        P0, P1, P2, P3 = P[i:i+4]
        
        # Compute the parameter t values
        t_values = np.linspace(0, 1, num_points)
        
        # Prepare the results array for this segment
        points = np.zeros((num_points, 2))
        
        for j, t in enumerate(t_values):
            T = np.array([t**3, t**2, t, 1])
            points[j] = T @ CR_matrix @ np.array([P0, P1, P2, P3])
        
        # Append the points to the results list
        all_points.append(points)
    
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

# Generate the Catmull-Rom spline curve
spline_points = catmull_rom_spline(control_points)

# Plot the result
plt.plot(spline_points[:, 0], spline_points[:, 1], label='Catmull-Rom Spline')
plt.scatter(control_points[:, 0], control_points[:, 1], color='red', label='Control Points')
plt.legend()
plt.show()
