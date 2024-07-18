import numpy as np
import matplotlib.pyplot as plt

def hermite_blend(t, tau=1.0):
    h1 = 2*t**3 - 3*t**2 + 1
    h2 = -2*t**3 + 3*t**2
    h3 = t**3 - 2*t**2 + t
    h4 = t**3 - t**2
    return h1, h2, tau * h3, tau * h4

def hermite_curve_with_tension(P0, P1, T0, T1, tau=1.0, num_points=100):
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 3))
    for i in range(num_points):
        h1, h2, h3, h4 = hermite_blend(t[i], tau)
        curve[i] = h1 * P0 + h2 * P1 + h3 * T0 + h4 * T1
    return curve

# Example points and tangents
P0 = np.array([0, 0, 0])
P1 = np.array([1, 1, 0])
T0 = np.array([1, 0, 0])
T1 = np.array([1, 0, 0])

# Generate curves with different tension values
tau_values = [0.5, 1.0, 2.0]
curves = [hermite_curve_with_tension(P0, P1, T0, T1, tau) for tau in tau_values]

# Plot the curves
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for curve, tau in zip(curves, tau_values):
    ax.plot(curve[:,0], curve[:,1], curve[:,2], label=f'Tension={tau}')
ax.scatter([P0[0], P1[0]], [P0[1], P1[1]], [P0[2], P1[2]], color='red')
ax.legend()
plt.show()
