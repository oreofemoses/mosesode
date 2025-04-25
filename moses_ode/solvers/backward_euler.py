import numpy as np
import logging
import sys #For float info
from scipy.optimize import approx_fprime

logger = logging.getLogger(__name__)

def newton_method(G, G_prime, y_init, tol=1e-6, max_iter=50):
    """
    Newton's method for solving G(y) = 0.
    Returns the solution and a convergence flag.
    """
    y = y_init.astype(float)
    for i in range(max_iter):
        G_val = G(y)
        if not np.isfinite(np.linalg.norm(G_val)):
            logger.warning("Non-finite G_val encountered, Newton's method diverging.")
            return y, False  # Stop if G_val is not finite
        if np.linalg.norm(G_val) < tol:
            logger.debug(f"Newton's method converged in {i+1} iterations.")
            return y, True
        G_prime_val = G_prime(y)
        delta_y = np.linalg.solve(G_prime_val, -G_val) if G_prime_val.ndim > 1 else -G_val / G_prime_val
        delta_y = np.asarray(delta_y) #add this line
        y += delta_y

        if np.linalg.norm(delta_y) < tol:
            logger.debug(f"Newton's method converged in {i+1} iterations.")
            return y, True

    logger.warning("Newton's method did not converge")
    return y, False  # Return the last iterate and a failure flag


def backward_euler_solver(f, t0, y0, h, t_end, jac=None, tol=1e-6, h_min=1e-12):
    """
    Backward Euler solver using Newton's method.
    """
    # Ensure y0 is a NumPy array
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    is_scalar = (y0.size == 1)  # Keep track if the ODE is scalar

    t_vals = [t0]
    y_vals = [y0]

    t = t0
    y = y0

    while t < t_end:
        t_next = t + h

        # Define the nonlinear function G(y_next) = y_next - y - h*f(t_next, y_next)
        def G(y_next):
            return y_next - y - h * f(t_next, y_next)

        # Compute Jacobian: G'(y_next) = I - h * df/dy
        def G_prime(y_next):
            if jac is not None:
                J = jac(t_next, y_next)  # Use user-supplied Jacobian
            else:
                J = compute_jacobian(f, t_next, y_next)  # Use f instead of G
            return np.eye(len(y)) - h * J if J.ndim > 1 else 1 - h * J

        # Solve for y_next using Newton's method
        y_next, converged = newton_method(G, G_prime, y, tol)

        if not converged:
            logger.warning(f"Newton's method failed to converge at t={t_next}, reducing step size.")
            h /= 2  # Reduce step size and retry
            if h < h_min:  # Check for minimum step size
                logger.error("Step size below minimum, stopping integration")
                break  # Exit the loop
            continue  # Retry the current time step with the smaller h

        # Store results
        t_vals.append(t_next)
        y_vals.append(y_next)

        # Advance to next step
        t = t_next
        y = y_next

    t_array = np.array(t_vals)
    y_array = np.array(y_vals)
    
    # If the original input was a scalar and the output is a column vector, flatten it
    if is_scalar and y_array.ndim > 1 and y_array.shape[1] == 1:
        y_array = y_array.flatten()
        
    return t_array, y_array


def compute_jacobian(f, t, y, eps=1e-6):
    """
    Computes the Jacobian matrix df/dy numerically using central finite differences.
    Handles both scalar and vector-valued functions.
    """
    y = np.asarray(y, dtype=float)

    if y.ndim == 0:  # Scalar case
        J = (f(t, y + eps) - f(t, y - eps)) / (2 * eps)
    else:  # Vector case
        J = np.zeros((len(y), len(y)))
        for i in range(len(y)):
            y_plus = y.copy()
            y_minus = y.copy()
            y_plus[i] += eps
            y_minus[i] -= eps
            J[:, i] = (f(t, y_plus) - f(t, y_minus)) / (2 * eps)

    return J