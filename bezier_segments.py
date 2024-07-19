import numpy as np

def bezier_curve(p, t):
    """
    Compute the value of the Bézier curve with control points p at time t.
    
    Parameters:
        p (numpy array): Control points of the curve.
        t (float): Time parameter.
        
    Returns:
        numpy array: Value of the curve at time t.
    """
    n = len(p) - 1
    return np.sum([np.power(1-t, n-i)*np.power(t, i)*p[i] for i in range(n+1)], axis=0)

def bezier_curve_derivative(p, t):
    """
    Compute the derivative of the Bézier curve with control points p at time t.
    
    Parameters:
        p (numpy array): Control points of the curve.
        t (float): Time parameter.
        
    Returns:
        numpy array: Derivative of the curve at time t.
    """
    n = len(p) - 1
    return np.sum([(n-i)*np.power(1-t, n-i-1)*(i+1)*np.power(t, i)*p[i] for i in range(n)], axis=0)

def connect_bezier_segments(p1, q1, q2, p2):
    """
    Compute the control points of a Bézier curve connecting two segments.
    
    Parameters:
        p1 (numpy array): Control points of the first segment.
        q1 (numpy array): Control points of the second segment.
        q2 (numpy array): Control points of the third segment.
        p2 (numpy array): Control points of the fourth segment.
        
    Returns:
        numpy array: Control points of the connected curve.
    """
    # Compute the intersection point of the tangents at the endpoints of the segments
    t1 = bezier_curve_derivative(p1, 1)
    t2 = bezier_curve_derivative(q1, 0)
    t3 = bezier_curve_derivative(q2, 1)
    t4 = bezier_curve_derivative(p2, 0)
    intersection = (t1 + t2)/(t1 - t2) * q1 + (t3 + t4)/(t3 - t4) * p2
    
    # Compute the control points of the connected curve
    c1 = (q1 + intersection)/2
    c2 = (intersection + p2)/2
    return np.vstack((c1, q2, c2))

# Example usage
p1 = np.array([[0, 0], [1, 0], [1, 1]])
q1 = np.array([[1, 1], [2, 1], [2, 2]])
q2 = np.array([[2, 2], [3, 2], [3, 3]])
p2 = np.array([[3, 3], [4, 3], [4, 4]])

# Compute the control points of the connected curve
control_points = connect_bezier_segments(p1, q1, q2, p2)

# Evaluate the curve at various times
times = np.linspace(0, 1, 100)
values = [bezier_curve(control_points, t) for t in times]

# Plot the curve
import matplotlib.pyplot as plt
plt.plot(values)
plt.show()

