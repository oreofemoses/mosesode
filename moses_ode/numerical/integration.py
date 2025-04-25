import numpy as np
import logging

logger = logging.getLogger(__name__)

def trapezoid(f, a, b, n=100):
    """
    Numerical integration using the Trapezoidal rule.
    
    Parameters:
    - f: Function to integrate
    - a: Lower bound
    - b: Upper bound
    - n: Number of subintervals
    
    Returns:
    - Approximation of the integral
    """
    logger.info(f"Applying Trapezoid rule with {n} subintervals")
    
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    
    for i in range(1, n):
        result += f(a + i * h)
    
    return h * result

def simpson_1_3(f, a, b, n=100):
    """
    Numerical integration using Simpson's 1/3 rule.
    
    Parameters:
    - f: Function to integrate
    - a: Lower bound
    - b: Upper bound
    - n: Number of subintervals (must be even)
    
    Returns:
    - Approximation of the integral
    """
    if n % 2 != 0:
        n += 1  # Ensure n is even
        
    logger.info(f"Applying Simpson's 1/3 rule with {n} subintervals")
    
    h = (b - a) / n
    result = f(a) + f(b)
    
    for i in range(1, n, 2):
        result += 4 * f(a + i * h)
        
    for i in range(2, n, 2):
        result += 2 * f(a + i * h)
    
    return h * result / 3

def simpson_3_8(f, a, b, n=99):
    """
    Numerical integration using Simpson's 3/8 rule.
    
    Parameters:
    - f: Function to integrate
    - a: Lower bound
    - b: Upper bound
    - n: Number of subintervals (must be multiple of 3)
    
    Returns:
    - Approximation of the integral
    """
    # Ensure n is a multiple of 3
    if n % 3 != 0:
        n += 3 - (n % 3)
        
    logger.info(f"Applying Simpson's 3/8 rule with {n} subintervals")
    
    h = (b - a) / n
    result = f(a) + f(b)
    
    for i in range(1, n):
        if i % 3 == 0:
            result += 2 * f(a + i * h)
        else:
            result += 3 * f(a + i * h)
    
    return 3 * h * result / 8

def romberg(f, a, b, max_iterations=10, tol=1e-10):
    """
    Numerical integration using Romberg method.
    
    Parameters:
    - f: Function to integrate
    - a: Lower bound
    - b: Upper bound
    - max_iterations: Maximum number of iterations
    - tol: Error tolerance
    
    Returns:
    - Approximation of the integral
    - R: Romberg table
    """
    logger.info(f"Applying Romberg integration with max iterations: {max_iterations}")
    
    # Input validation
    if not callable(f):
        logger.error("Input function f is not callable")
        raise TypeError("Input function f must be callable")
    
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        logger.error(f"Invalid bounds: a={a}, b={b}")
        raise TypeError("Lower and upper bounds must be numeric values")
    
    if a >= b:
        logger.warning(f"Lower bound {a} >= upper bound {b}, swapping bounds")
        a, b = b, a
    
    # Initialize R matrix
    R = np.zeros((max_iterations, max_iterations))
    h = b - a
    
    # Try to evaluate the function at endpoints to catch errors early
    try:
        fa = f(a)
        fb = f(b)
    except Exception as e:
        logger.error(f"Error evaluating function at endpoints: {str(e)}")
        raise ValueError(f"Failed to evaluate function at integration bounds: {str(e)}")
    
    # First approximation with trapezoidal rule
    R[0, 0] = h * (fa + fb) / 2
    
    try:
        for i in range(1, max_iterations):
            # Compute next trapezoidal approximation
            h = h / 2
            sum_f = 0
            for k in range(1, 2**i, 2):
                try:
                    sum_f += f(a + k * h)
                except Exception as e:
                    logger.error(f"Error evaluating function at x={a + k * h}: {str(e)}")
                    raise ValueError(f"Failed to evaluate function during integration: {str(e)}")
            
            R[i, 0] = R[i-1, 0] / 2 + h * sum_f
            
            # Compute higher-order approximations using Richardson extrapolation
            for j in range(1, i+1):
                R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4**j - 1)
                
            # Check for convergence
            if i > 0 and abs(R[i, i] - R[i-1, i-1]) < tol:
                logger.info(f"Romberg integration converged after {i+1} iterations")
                return R[i, i], R[:i+1, :i+1]
    
        logger.warning("Romberg integration did not converge to desired tolerance")
        return R[max_iterations-1, max_iterations-1], R
    except Exception as e:
        logger.error(f"Unexpected error in Romberg integration: {str(e)}")
        # Return best estimate and table so far instead of None
        return R[max(0, max_iterations-1)][max(0, max_iterations-1)], R 