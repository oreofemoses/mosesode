import numpy as np
from .backward_euler import newton_method  # Custom Newton solver
from .backward_euler import compute_jacobian  # Numerical Jacobian computation

def crank_nicolson_solver(f, t0, y0, h, t_end, tol=1e-6, max_iter=50, jac=None):
    """
    Fixed-step Crank-Nicolson solver for ODEs using a custom Newton's method.
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

        # Define the nonlinear function G(y_next) = 0
        def G(y_next):
            return y_next - y - 0.5 * h * (f(t, y) + f(t_next, y_next))

        # Define Jacobian of G (dG/dy_next)
        def G_prime(y_next):
            if jac is not None:
                J = jac(t_next, y_next)  # Use analytical Jacobian if provided
            else:
                J = compute_jacobian(f, t_next, y_next)  # Numerical Jacobian
            return np.eye(len(y)) - 0.5 * h * J if J.ndim > 1 else 1 - 0.5 * h * J

        # Solve for y_next using our custom Newton's method
        y_next, converged = newton_method(G, G_prime, y, tol, max_iter)

        if not converged:
            raise RuntimeError(f"Newton's method failed to converge at t={t_next}")

        # Store results
        t_vals.append(t_next)
        y_vals.append(y_next)
        t, y = t_next, y_next

    t_array = np.array(t_vals)
    y_array = np.array(y_vals)
    
    # If the original input was a scalar and the output is a column vector, flatten it
    if is_scalar and y_array.ndim > 1 and y_array.shape[1] == 1:
        y_array = y_array.flatten()
        
    return t_array, y_array