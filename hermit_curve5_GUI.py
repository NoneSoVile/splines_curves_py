import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from mpl_toolkits.mplot3d import Axes3D

# Define the degree-5 Hermite blending functions
def hermite_blend(t):
    H1 = 1 - 10*t**3 + 15*t**4 - 6*t**5
    H2 = t - 6*t**3 + 8*t**4 - 3*t**5
    H3 = 0.5 * t**2 - 3*t**3 + 3.5*t**4 - 2*t**5
    H4 = 10*t**3 - 15*t**4 + 6*t**5
    H5 = -4*t**3 + 7*t**4 - 3*t**5
    H6 = 1.5*t**3 - 3.5*t**4 + 2*t**5
    return H1, H2, H3, H4, H5, H6

# Construct the degree-5 Hermite curve
def degree_5_hermite_curve(P0, P1, T0, T1, D0, D1, num_points=100):
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 3))
    for i in range(num_points):
        H1, H2, H3, H4, H5, H6 = hermite_blend(t[i])
        curve[i] = H1 * P0 + H2 * T0 + H3 * D0 + H4 * P1 + H5 * T1 + H6 * D1
    return curve

# Update function for interactive plotting
def update(px0, py0, pz0, px1, py1, pz1, tx0, ty0, tz0, tx1, ty1, tz1, dx0, dy0, dz0, dx1, dy1, dz1):
    P0 = np.array([px0, py0, pz0])
    P1 = np.array([px1, py1, pz1])
    T0 = np.array([tx0, ty0, tz0])
    T1 = np.array([tx1, ty1, tz1])
    D0 = np.array([dx0, dy0, dz0])
    D1 = np.array([dx1, dy1, dz1])
    
    curve = degree_5_hermite_curve(P0, P1, T0, T1, D0, D1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(curve[:,0], curve[:,1], curve[:,2], label='Degree-5 Hermite Curve')
    ax.scatter([P0[0], P1[0]], [P0[1], P1[1]], [P0[2], P1[2]], color='red', label='Control Points')
    ax.legend()
    plt.show()

# Interactive sliders
interact(update,
         px0=FloatSlider(min=-5, max=5, step=0.1, value=0, description='P0 x'),
         py0=FloatSlider(min=-5, max=5, step=0.1, value=0, description='P0 y'),
         pz0=FloatSlider(min=-5, max=5, step=0.1, value=0, description='P0 z'),
         px1=FloatSlider(min=-5, max=5, step=0.1, value=1, description='P1 x'),
         py1=FloatSlider(min=-5, max=5, step=0.1, value=2, description='P1 y'),
         pz1=FloatSlider(min=-5, max=5, step=0.1, value=0, description='P1 z'),
         tx0=FloatSlider(min=-5, max=5, step=0.1, value=1, description='T0 x'),
         ty0=FloatSlider(min=-5, max=5, step=0.1, value=0, description='T0 y'),
         tz0=FloatSlider(min=-5, max=5, step=0.1, value=0, description='T0 z'),
         tx1=FloatSlider(min=-5, max=5, step=0.1, value=1, description='T1 x'),
         ty1=FloatSlider(min=-5, max=5, step=0.1, value=1, description='T1 y'),
         tz1=FloatSlider(min=-5, max=5, step=0.1, value=0, description='T1 z'),
         dx0=FloatSlider(min=-5, max=5, step=0.1, value=0, description='D0 x'),
         dy0=FloatSlider(min=-5, max=5, step=0.1, value=1, description='D0 y'),
         dz0=FloatSlider(min=-5, max=5, step=0.1, value=0, description='D0 z'),
         dx1=FloatSlider(min=-5, max=5, step=0.1, value=0, description='D1 x'),
         dy1=FloatSlider(min=-5, max=5, step=0.1, value=-1, description='D1 y'),
         dz1=FloatSlider(min=-5, max=5, step=0.1, value=0, description='D1 z'))
