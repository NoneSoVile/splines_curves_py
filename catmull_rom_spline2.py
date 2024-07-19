import numpy as np
import matplotlib.pyplot as plt

def parabolic_blend_matrix():
    return np.array([
        [-0.5,  1.5, -1.5,  0.5],
        [ 1.0, -2.5,  2.0, -0.5],
        [-0.5,  0.0,  0.5,  0.0],
        [ 0.0,  1.0,  0.0,  0.0]
    ])

def compute_H(P1, P2, P3):
    M_inv = np.linalg.inv(np.array([
        [ 1,  0,  0],
        [ 0,  1,  0],
        [ 0,  0,  1]
    ]))
    return M_inv @ np.vstack([P1, P2, P3]).T

def catmull_rom_spline(P0, P1, P2, P3, t):
    # Calculate parabolic blending matrices
    H123 = compute_H(P0, P1, P2)
    H234 = compute_H(P1, P2, P3)
    
    # Calculate blending functions
    u = np.array([t**2, t, 1])
    w = np.array([t**2, t, 1])
    Q = u @ H123
    R = w @ H234
    
    return (1 - t) * Q + t * R

def catmull_rom_chain(P):
    points = []
    for i in range(len(P) - 3):
        for t in np.linspace(0, 1, 100):
            points.append(catmull_rom_spline(P[i], P[i+1], P[i+2], P[i+3], t))
    return np.array(points)

# Example usage
P = np.array([[0, 0], [1, 0], [3, 1], [6, 2], [2, 3], [1, 4]])

curve_points = catmull_rom_chain(P)

plt.plot(P[:, 0], P[:, 1], 'ro-')  # Control points
plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-')  # Curve
plt.show()
