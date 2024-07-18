import numpy as np
import matplotlib.pyplot as plt

# Given points
P1 = np.array([0, 0])
P2 = np.array([1, 0])
P3 = np.array([1, 1])
P4 = np.array([0, 1])

# Given tangent vectors
P1_t = np.array([1, -1])
P4_t = np.array([-1, -1])

# Store the points and tangents in arrays
points = [P1, P2, P3, P4]
tangents = [P1_t, None, None, P4_t]  # Initial guesses for internal tangents are None

def hermite_basis(t):
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    return h00, h10, h01, h11

def hermite_cubic(P0, P1, P0_t, P1_t, t):
    h00, h10, h01, h11 = hermite_basis(t)
    return h00*P0 + h10*P0_t + h01*P1 + h11*P1_t

# Define second derivative of Hermite basis functions
def hermite_basis_second_derivative(t):
    h00_ddot = 12*t - 6
    h10_ddot = 6*t - 4
    h01_ddot = -12*t + 6
    h11_ddot = 6*t - 2
    return h00_ddot, h10_ddot, h01_ddot, h11_ddot

from scipy.optimize import fsolve

def matching_second_derivatives(tangent_guesses):
    P2_t = tangent_guesses[:2]
    P3_t = tangent_guesses[2:]
    
    # Second derivative at P2 from segment P1(t)
    t = 1
    h00_ddot, h10_ddot, h01_ddot, h11_ddot = hermite_basis_second_derivative(t)
    second_deriv_1 = h00_ddot*P1 + h10_ddot*P1_t + h01_ddot*P2 + h11_ddot*P2_t
    
    # Second derivative at P2 from segment P2(t)
    t = 0
    second_deriv_2 = h00_ddot*P2 + h10_ddot*P2_t + h01_ddot*P3 + h11_ddot*P3_t
    
    # Second derivative at P3 from segment P2(t)
    t = 1
    h00_ddot, h10_ddot, h01_ddot, h11_ddot = hermite_basis_second_derivative(t)
    second_deriv_3 = h00_ddot*P2 + h10_ddot*P2_t + h01_ddot*P3 + h11_ddot*P3_t
    
    # Second derivative at P3 from segment P3(t)
    t = 0
    second_deriv_4 = h00_ddot*P3 + h10_ddot*P3_t + h01_ddot*P4 + h11_ddot*P4_t
    
    return np.concatenate((second_deriv_1 - second_deriv_2, second_deriv_3 - second_deriv_4))

# Initial guesses for the internal tangents
initial_guesses = np.array([0, 0, 0, 0])

# Solve for the internal tangents
solution = fsolve(matching_second_derivatives, initial_guesses)
P2_t = solution[:2]
P3_t = solution[2:]

# Update the tangents array with the solved values
tangents[1] = P2_t
tangents[2] = P3_t

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

