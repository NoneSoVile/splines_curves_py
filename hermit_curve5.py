import numpy as np
import matplotlib.pyplot as plt

# Define the degree-5 Hermite blending functions
def hermite_blend(t):
    F1 = 1 - 10*t**3 + 15*t**4 - 6*t**5
    F2 = t - 6*t**3 + 8*t**4 - 3*t**5
    F3 = 0.5 * t**2 - 3*t**3 + 3.5*t**4 - 2*t**5
    F4 = 10*t**3 - 15*t**4 + 6*t**5
    F5 = -4*t**3 + 7*t**4 - 3*t**5
    F6 = 1.5*t**3 - 3.5*t**4 + 2*t**5
    return F1, F2, F3, F4, F5, F6

# Construct the degree-5 Hermite curve
def degree_5_hermite_curve(P0, P1, P2, T0, T1, T2, num_points=100):
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 3))
    for i in range(num_points):
        F1, F2, F3, F4, F5, F6 = hermite_blend(t[i])
        curve[i] = F1 * P0 + F2 * T0 + F3 * T1 + F4 * P1 + F5 * T1 + F6 * P2
    return curve

# Example points and tangents
P0 = np.array([0, 0, 0])
P1 = np.array([1, 2, 0])
P2 = np.array([2, 0, 0])
T0 = np.array([1, 0, 0])
T1 = np.array([1, 1, 0])
T2 = np.array([0, -1, 0])

# Generate the curve
curve = degree_5_hermite_curve(P0, P1, P2, T0, T1, T2)

# Plot the curve
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(curve[:,0], curve[:,1], curve[:,2], label='Degree-5 Hermite Curve')
ax.scatter([P0[0], P1[0], P2[0]], [P0[1], P1[1], P2[1]], [P0[2], P1[2], P2[2]], color='red', label='Control Points')
ax.legend()
plt.show()
