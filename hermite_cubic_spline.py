import numpy as np
import matplotlib.pyplot as plt

# Define the Hermite basis functions
def h00(t):
    return 2 * t**3 - 3 * t**2 + 1

def h10(t):
    return t**3 - 2 * t**2 + t

def h01(t):
    return -2 * t**3 + 3 * t**2

def h11(t):
    return t**3 - t**2

# Function to compute the Hermite cubic polynomial
def hermite_cubic(p0, p1, m0, m1, t):
    return h00(t) * p0 + h10(t) * m0 + h01(t) * p1 + h11(t) * m1

# Define the data points
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([0, 1, 3, 5, 4, 5, 6, 7, 3, 1])

# Compute tangents (for simplicity, using finite differences)
m = np.zeros_like(y)
m[0] = (y[1] - y[0]) / (x[1] - x[0])
m[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
for i in range(1, len(y) - 1):
    m[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])

# Evaluate the Hermite spline
x_new = np.linspace(x[0], x[-1], 400)
y_new = np.zeros_like(x_new)

for i in range(len(x) - 1):
    xi = x[i]
    xi1 = x[i + 1]
    for j in range(len(x_new)):
        if xi <= x_new[j] <= xi1:
            t = (x_new[j] - xi) / (xi1 - xi)
            y_new[j] = hermite_cubic(y[i], y[i + 1], m[i] * (xi1 - xi), m[i + 1] * (xi1 - xi), t)

# Plot the result
plt.plot(x, y, 'o', label='Data points')
plt.plot(x_new, y_new, '-', label='Cubic Hermite Spline')
plt.legend()
plt.show()
