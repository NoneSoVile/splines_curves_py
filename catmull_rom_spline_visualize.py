import numpy as np
import matplotlib.pyplot as plt

def parabola_through_3_points(P0, P1, P2):
    x0, y0 = P0
    x1, y1 = P1
    x2, y2 = P2
    
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2*x2 * (y0 - y1) + x1*x1 * (y2 - y0) + x0*x0 * (y1 - y2)) / denom
    c = (x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1 + x0 * x1 * (x0 - x1) * y2) / denom
    
    return a, b, c

def catmull_rom_segment(P0, P1, P2, P3, num_points=100):
    a1, b1, c1 = parabola_through_3_points(P0, P1, P2)
    a2, b2, c2 = parabola_through_3_points(P1, P2, P3)
    
    t = np.linspace(0, 1, num_points)
    
    # Calculate points for each parabola
    x1 = a1 * t**2 + b1 * t + c1
    y1 = (1 - t) * P1[1] + t * P2[1]
    
    x2 = a2 * t**2 + b2 * t + c2
    y2 = (1 - t) * P1[1] + t * P2[1]
    
    # Blend the two parabolas
    x_blend = (1 - t) * x1 + t * x2
    y_blend = y1  # y values are the same for both parabolas in this implementation
    
    return (x1, y1), (x2, y2), (x_blend, y_blend)

# Example usage
P0, P1, P2, P3 = (0, 0), (1, 5), (4, 4), (7, 6)

# Create the spline segment
(x1, y1), (x2, y2), (x_blend, y_blend) = catmull_rom_segment(P0, P1, P2, P3)

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(*zip(P0, P1, P2, P3), 'ro-', label='Control Points')
plt.plot(x1, y1, 'g--', label='Parabola 1')
plt.plot(x2, y2, 'b--', label='Parabola 2')
plt.plot(x_blend, y_blend, 'k-', linewidth=2, label='Blended Curve')
plt.legend()
plt.title('Catmull-Rom Spline: Blended Parabolas Visualization')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()