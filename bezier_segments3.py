import matplotlib.pyplot as plt
import numpy as np

def bezier_curve(points, num_points=100):
    n = len(points) - 1
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2))
    for i in range(num_points):
        for j in range(n + 1):
            bernstein = (np.math.factorial(n) /
                         (np.math.factorial(j) * np.math.factorial(n - j))) * \
                        (t[i] ** j) * ((1 - t[i]) ** (n - j))
            curve[i] += bernstein * np.array(points[j])
    return curve

def adjust_control_points_for_tangents(segments):
    adjusted_segments = [segments[0]]
    for i in range(1, len(segments)):
        prev_segment = adjusted_segments[-1]
        curr_segment = segments[i]
        # Convert points to numpy arrays
        prev_end = np.array(prev_segment[-1])
        prev_cp = np.array(prev_segment[-2])
        curr_start = np.array(curr_segment[0])
        
        # Vector from the previous control point to the previous end point
        tangent = prev_end - prev_cp
        
        # Adjust the first control point of the current segment to align tangents
        new_first_cp = curr_start + tangent
        
        adjusted_segment = [curr_start.tolist(), new_first_cp.tolist()] + curr_segment[1:]
        adjusted_segments.append(adjusted_segment)
    
    return adjusted_segments

def draw_bezier_segments(segments, colors):
    for points, color in zip(segments, colors):
        curve = bezier_curve(points)
        plt.plot(curve[:, 0], curve[:, 1], color=color)
        control_points = np.array(points)
        plt.plot(control_points[:, 0], control_points[:, 1], 'o', color=color)
        for i in range(len(points) - 1):
            plt.plot(control_points[i:i+2, 0], control_points[i:i+2, 1], '--', color=color)

# Define control points for each segment
segments = [
    [(0, 0), (1, 2), (3, 3)],
    [(3, 3), (4, 4), (5, 2), (6, 1)],
    [(6, 1), (7, 1), (8, 0)],
    [(8, 0), (9, 1), (12, 2)],
]

# Adjust control points to ensure C1 continuity
adjusted_segments = adjust_control_points_for_tangents(segments)

# Define colors for each segment
colors = ['red', 'green', 'blue', 'red',]

plt.figure()
draw_bezier_segments(adjusted_segments, colors)
plt.title('Bezier Curve Segments with Control Points')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
