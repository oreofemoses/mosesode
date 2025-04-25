import numpy as np
import logging
from moses_ode.validation.input_validation import validate_function_input

logger = logging.getLogger(__name__)


def heun_solver(f, t0, y0, h, t_end):
    """
    Solves an ODE using Heun's Method (Improved Euler / RK2 variant).

    Parameters:
        f (callable): The ODE function f(t, y).
        t0 (float): Initial time.
        y0 (float or np.ndarray): Initial value(s).
        h (float): Step size.
        t_end (float): Final time.

    Returns:
        t_values (np.ndarray): Time steps.
        y_values (np.ndarray): Solution values at each step.

    Raises:
        ValueError: If h <= 0 or t_end <= t0.
    """
    # Input validation
    if h <= 0:
        raise ValueError("Step size h must be positive.")
    if t_end <= t0:
        raise ValueError("End time t_end must be greater than initial time t0.")
    f = validate_function_input(f, t0, y0)

    logger.info(f"Solving ODE using Heun's method from t={t0} to t={t_end} with step size h={h}.")

    # Compute number of steps ensuring final step reaches t_end
    num_steps = int(np.ceil((t_end - t0) / h))
    t_values = np.linspace(t0, t0 + num_steps * h, num_steps + 1)

    # Initialize solution array
    y_values = np.zeros((len(t_values), np.size(y0)))
    y_values[0] = y0

    # Heun's method iteration
    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        y = y_values[i - 1]

        # Predictor step (Euler's estimate)
        y_pred = y + h * f(t, y)

        # Corrector step (Averaged slope)
        y_values[i] = y + (h / 2) * (f(t, y) + f(t_values[i], y_pred))

        logger.debug(f"Step {i}: t={t_values[i]:.4f}, y={y_values[i]}")

    # Convert to 1D array if y0 was scalar
    if np.isscalar(y0):
        y_values = y_values.flatten()

    logger.info("Heun's solver completed successfully.")
    return t_values, y_values

