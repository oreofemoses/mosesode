import numpy as np
import logging

logger = logging.getLogger(__name__)

def forward_difference(f, x, h=1e-5):
    """
    First-order forward difference approximation of the derivative.
    
    Parameters:
    - f: Function to differentiate
    - x: Point at which to calculate the derivative
    - h: Step size
    
    Returns:
    - Approximation of the derivative
    """
    logger.info(f"Calculating forward difference at x={x} with h={h}")
    return (f(x + h) - f(x)) / h

def backward_difference(f, x, h=1e-5):
    """
    First-order backward difference approximation of the derivative.
    
    Parameters:
    - f: Function to differentiate
    - x: Point at which to calculate the derivative
    - h: Step size
    
    Returns:
    - Approximation of the derivative
    """
    logger.info(f"Calculating backward difference at x={x} with h={h}")
    return (f(x) - f(x - h)) / h

def central_difference(f, x, h=1e-5):
    """
    Second-order central difference approximation of the derivative.
    
    Parameters:
    - f: Function to differentiate
    - x: Point at which to calculate the derivative
    - h: Step size
    
    Returns:
    - Approximation of the derivative
    """
    logger.info(f"Calculating central difference at x={x} with h={h}")
    return (f(x + h) - f(x - h)) / (2 * h)

def second_derivative(f, x, h=1e-4):
    """
    Second-order approximation of the second derivative.
    
    Parameters:
    - f: Function to differentiate
    - x: Point at which to calculate the derivative
    - h: Step size
    
    Returns:
    - Approximation of the second derivative
    """
    logger.info(f"Calculating second derivative at x={x} with h={h}")
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)

def divided_difference(f, x_values):
    """
    Calculate divided differences for a set of points.
    
    Parameters:
    - f: Function or array of function values at x_values
    - x_values: Array of x points
    
    Returns:
    - Table of divided differences
    """
    n = len(x_values)
    
    # Check if f is a function or array of values
    if callable(f):
        # If f is a function, evaluate it at each x
        f_values = np.array([f(x) for x in x_values])
    else:
        # If f is already an array of values
        f_values = np.array(f)
    
    # Create a table for divided differences
    table = np.zeros((n, n))
    
    # First column is function values
    table[:, 0] = f_values
    
    # Calculate divided differences
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x_values[i + j] - x_values[i])
    
    logger.info(f"Calculated divided difference table of size {n}x{n}")
    return table

def newton_polynomial(divided_diff_table, x_values, x):
    """
    Evaluate Newton's interpolation polynomial at point x.
    
    Parameters:
    - divided_diff_table: Table of divided differences
    - x_values: Array of x points used to create the table
    - x: Point at which to evaluate the polynomial
    
    Returns:
    - Interpolated value at x
    """
    n = len(x_values)
    result = divided_diff_table[0, 0]
    
    # Build polynomial term by term
    term = 1.0
    for i in range(1, n):
        term *= (x - x_values[i-1])
        result += divided_diff_table[0, i] * term
    
    return result

def derivative_from_divided_diff(divided_diff_table, x_values, order=1):
    """
    Calculate derivative of order 'order' at x_values[0] using divided differences.
    
    Parameters:
    - divided_diff_table: Table of divided differences
    - x_values: Array of x points used to create the table
    - order: Order of the derivative (1 for first derivative, etc.)
    
    Returns:
    - Approximation of the derivative at x_values[0]
    """
    if order > len(x_values) - 1:
        raise ValueError(f"Order {order} too high for {len(x_values)} points")
    
    # Factorial calculation
    factorial = 1
    for i in range(1, order + 1):
        factorial *= i
    
    logger.info(f"Calculating {order}-th derivative from divided differences")
    return factorial * divided_diff_table[0, order] 