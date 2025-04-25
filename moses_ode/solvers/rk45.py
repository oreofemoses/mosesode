import numpy as np
import logging
from moses_ode.validation.input_validation import validate_function_input

logger = logging.getLogger(__name__)


def rk45_solver(f, t0, y0, h_init, t_end, tol=1e-6, h_min=1e-12, h_max=1.0):
    """
    Solves an ODE using the Adaptive Runge-Kutta-Fehlberg (RK45) method.
    """
    if h_init <= 0:
        raise ValueError("Initial step size must be positive.")
    if t_end <= t0:
        raise ValueError("End time must be greater than initial time.")

    f = validate_function_input(f, t0, y0)  # Validate input first

    logger.info(f"Solving ODE using RK45 from t={t0} to t={t_end} with initial step size h={h_init}.")

    # Butcher tableau coefficients for RK45
    a = np.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2])
    b = np.array([
        [0, 0, 0, 0, 0],
        [1 / 4, 0, 0, 0, 0],
        [3 / 32, 9 / 32, 0, 0, 0],
        [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0],
        [439 / 216, -8, 3680 / 513, -845 / 4104, 0],
        [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40]
    ])
    c4 = np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])  # 4th order
    c5 = np.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])  # 5th order

    t_values, y_values = [t0], [np.array(y0, dtype=float)]
    t, y, h = t0, np.array(y0, dtype=float), h_init

    while t < t_end:
        if t + h > t_end:
            h = t_end - t  # Adjust final step size

        k = np.zeros((6, y.shape[0])) if y.ndim > 0 else np.zeros(6)  # Handle vector/scalar cases
        for i in range(6):
            k[i] = h * f(t + a[i] * h, y + np.dot(b[i, :i], k[:i]))

        y4 = y + np.dot(c4, k)  # 4th-order solution
        y5 = y + np.dot(c5, k)  # 5th-order solution

        # error = np.linalg.norm(y5 - y4) / (np.linalg.norm(y5) + 1e-8)  # Relative error
        # Use mixed absolute/relative tolerance scale
        atol = 1e-6 # Fixed absolute tolerance component
        scale = atol + tol * max(np.linalg.norm(y), np.linalg.norm(y5))
        error = np.linalg.norm(y5 - y4) / (scale + 1e-10) # Add epsilon for safety

        if error < 1.0: # Error is relative to tolerance, check if < 1
            t += h
            y = y5  # Accept the step
            t_values.append(t)
            y_values.append(y.copy())

        # Compute optimal step size
        # h_new = 0.84 * h * (tol / (error + 1e-8)) ** 0.25
        # Corrected formula with safety factor and exponent 1/5=0.2
        h_new = 0.9 * h * (1.0 / (error + 1e-10)) ** 0.2 # Use 1/error since error is now scaled by tol
        h = min(max(h_new, 0.1 * h), 5 * h, h_max)

        if h < h_min:
            logger.warning(f"Step size {h} below minimum {h_min}, stopping integration.")
            break

    logger.info("RK45 solver completed successfully.")
    return np.array(t_values), np.array(y_values)

