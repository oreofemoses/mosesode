import numpy as np
import logging
from moses_ode.numerical.differentiation import (
    forward_difference,
    central_difference,
    second_derivative
)
from moses_ode.numerical.integration import trapezoid, simpson_1_3

logger = logging.getLogger(__name__)

def estimate_ode_coefficients(data_points, order=1):
    """
    Estimate the coefficients of an ODE from data points.
    
    Parameters:
    - data_points: List of (t, y) tuples representing data measurements
    - order: Order of the ODE to fit (1 or 2)
    
    Returns:
    - dictionary of estimated coefficients
    """
    if order not in [1, 2]:
        raise ValueError("Only first and second order ODEs are supported")
    
    t_values = np.array([point[0] for point in data_points])
    y_values = np.array([point[1] for point in data_points])
    
    # Define interpolation function using numpy's interp
    def y_interp(t):
        return np.interp(t, t_values, y_values)
    
    # Calculate derivatives at each point
    dy_dt_values = []
    d2y_dt2_values = [] if order == 2 else None
    
    for t in t_values:
        dy_dt_values.append(central_difference(y_interp, t, h=(t_values[-1] - t_values[0])/1000))
        if order == 2:
            d2y_dt2_values.append(second_derivative(y_interp, t, h=(t_values[-1] - t_values[0])/100))
    
    # For first-order ODE: dy/dt = a*y + b
    if order == 1:
        # Solve for a and b using least squares
        A = np.vstack([y_values, np.ones(len(t_values))]).T
        a, b = np.linalg.lstsq(A, dy_dt_values, rcond=None)[0]
        return {'a': a, 'b': b}
    
    # For second-order ODE: d²y/dt² = a*y + b*dy/dt + c
    else:
        A = np.vstack([y_values, np.array(dy_dt_values), np.ones(len(t_values))]).T
        a, b, c = np.linalg.lstsq(A, d2y_dt2_values, rcond=None)[0]
        return {'a': a, 'b': b, 'c': c}

def integrate_function_to_solve_ode(f, a, b, y0, n=1000, method='simpson'):
    """
    Solve simple first-order separable ODE of the form dy/dt = f(t) by integration.
    
    Parameters:
    - f: Right-hand side function f(t)
    - a: Initial time
    - b: Final time
    - y0: Initial condition y(a) = y0
    - n: Number of steps
    - method: Integration method ('trapezoid' or 'simpson')
    
    Returns:
    - Tuple of time points and solution values
    """
    logger.info(f"Solving ODE dy/dt = f(t) with {method} integration")
    
    h = (b - a) / n
    t_points = np.linspace(a, b, n + 1)
    y_values = np.zeros(n + 1)
    y_values[0] = y0
    
    for i in range(1, n + 1):
        # Integrate from a to t_i
        t_i = t_points[i]
        
        if method == 'trapezoid':
            integral = trapezoid(f, a, t_i, n=i)
        elif method == 'simpson':
            # Ensure even number of subintervals
            n_i = i if i % 2 == 0 else i + 1
            integral = simpson_1_3(f, a, t_i, n=n_i)
        else:
            raise ValueError("Method must be 'trapezoid' or 'simpson'")
        
        y_values[i] = y0 + integral
    
    return t_points, y_values

def estimate_ode_from_solution(t_points, y_values, order=1):
    """
    Given a numerical solution, estimate the ODE that generated it.
    
    Parameters:
    - t_points: Array of time points
    - y_values: Array of y values corresponding to t_points
    - order: Assumed order of the ODE (1 or 2)
    
    Returns:
    - Dictionary with estimated ODE coefficients and equation string
    """
    # Define interpolation function
    def y_interp(t):
        return np.interp(t, t_points, y_values)
    
    # Sample points (using a subset to reduce noise impact)
    n = len(t_points)
    sample_indices = np.linspace(0, n-1, min(20, n), dtype=int)
    
    data_points = [(t_points[i], y_values[i]) for i in sample_indices]
    coeffs = estimate_ode_coefficients(data_points, order=order)
    
    if order == 1:
        equation = f"dy/dt = {coeffs['a']:.4f}*y + {coeffs['b']:.4f}"
    else:
        equation = f"d²y/dt² = {coeffs['a']:.4f}*y + {coeffs['b']:.4f}*dy/dt + {coeffs['c']:.4f}"
    
    return {**coeffs, 'equation': equation} 