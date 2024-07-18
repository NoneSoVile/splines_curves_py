import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def hermite_basis(t):
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    return h00, h10, h01, h11

def hermite_cubic(P0, P1, P0_t, P1_t, t):
    h00, h10, h01, h11 = hermite_basis(t)
    return h00*P0 + h10*P0_t + h01*P1 + h11*P1_t

P1 = np.array([0, 0])
P2 = np.array([1, 0])
P3 = np.array([1, 1])
P4 = np.array([0, 1])
P1_t = np.array([1, -1])
P4_t = np.array([-1, -1])

def tangent_equations(tangent_guesses):
    P2_t = tangent_guesses[:2]
    P3_t = tangent_guesses[2:]
    
    # Equations for the tangent vectors
    eq1 = hermite_cubic(P1, P2, P1_t, P2_t, 1) - P2
    eq2 = hermite_cubic(P2, P3, P2_t, P3_t, 0) - P2
    eq3 = hermite_cubic(P2, P3, P2_t, P3_t, 1) - P3
    eq4 = hermite_cubic(P3, P4, P3_t, P4_t, 0) - P3
    
    return np.concatenate((P2_t, P3_t))

# Initial guesses for the internal tangents
initial_guesses = np.array([0.1, 0.1, -0.1, 0.1])

# Solve for the internal tangents
solution = fsolve(tangent_equations, initial_guesses)
P2_t = solution[:2]
P3_t = solution[2:]

# Create t values
t_values = np.linspace(0, 1, 100)

# Hermite segments
segment1 = np.array([hermite_cubic(P1, P2, P1_t, P2_t, t) for t in t_values])
segment2 = np.array([hermite_cubic(P2, P3, P2_t, P3_t, t) for t in t_values])
segment3 = np.array([hermite_cubic(P3, P4, P3_t, P4_t, t) for t in t_values])

# Plotting the result
plt.figure(figsize=(8, 8))
plt.plot(segment1[:, 0], segment1[:, 1], label='Segment 1')
plt.plot(segment2[:, 0], segment2[:, 1], label='Segment 2')
plt.plot(segment3[:, 0], segment3[:, 1], label='Segment 3')
plt.scatter([P1[0], P2[0], P3[0], P4[0]], [P1[1], P2[1], P3[1], P4[1]], color='red')
plt.quiver(*P1, *P1_t, color='green', angles='xy', scale_units='xy', scale=1)
plt.quiver(*P2, *P2_t, color='blue', angles='xy', scale_units='xy', scale=1)
plt.quiver(*P3, *P3_t, color='blue', angles='xy', scale_units='xy', scale=1)
plt.quiver(*P4, *P4_t, color='green', angles='xy', scale_units='xy', scale=1)
plt.text(P1[0], P1[1], 'P1', fontsize=12, ha='right')
plt.text(P2[0], P2[1], 'P2', fontsize=12, ha='left')
plt.text(P3[0], P3[1], 'P3', fontsize=12, ha='left')
plt.text(P4[0], P4[1], 'P4', fontsize=12, ha='right')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.show()
