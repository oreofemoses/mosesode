import numpy as np
import logging
from moses_ode.validation.input_validation import validate_function_input

logger = logging.getLogger(__name__)


def rk4_solver(f, t0, y0, h, t_end):
    """
    Solves an ODE using the 4th-order Runge-Kutta (RK4) method.

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

    logger.info(f"Solving ODE using RK4 from t={t0} to t={t_end} with step size h={h}.")

    # Compute number of steps ensuring final step reaches t_end
    num_steps = int(np.ceil((t_end - t0) / h))
    t_values = np.linspace(t0, t0 + num_steps * h, num_steps + 1)

    # Initialize solution array
    y_values = np.zeros((len(t_values), np.size(y0)))
    y_values[0] = y0

    # RK4 iteration
    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        y = y_values[i - 1]

        k1 = h * f(t, y)
        k2 = h * f(t + h / 2, y + k1 / 2)
        k3 = h * f(t + h / 2, y + k2 / 2)
        k4 = h * f(t + h, y + k3)

        y_values[i] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        logger.debug(f"Step {i}: t={t_values[i]:.4f}, y={y_values[i]}")

    # Convert to 1D array if y0 was scalar
    if np.isscalar(y0):
        y_values = y_values.flatten()
    elif y_values.ndim == 2 and y_values.shape[1] == 1:
        y_values = y_values.reshape(-1)  # Ensure (N,1) becomes (N,)

    logger.info("RK4 solver completed successfully.")
    return t_values, y_values

# import numpy as np
# import logging
# from moses_ode.validation.input_validation import validate_function_input, validate_step_size, validate_time_span, validate_initial_conditions
# from moses_ode.solvers.solver_base import ODESolver
#
# logger = logging.getLogger(__name__)
#
#
# class RK4Solver(ODESolver):
#     """
#     Solves an ODE using the 4th-order Runge-Kutta (RK4) method.
#     """
#
#     config = {
#         "f": {"type": callable, "required": True, "description": "The ODE function f(t, y)"},
#         "t_span": {"type": (tuple, list), "required": True, "description": "(t0, tf) integration interval"},
#         "y0": {"type": (int, float, np.ndarray), "required": True, "description": "Initial condition"},
#         "h": {"type": float, "required": True, "description": "Step size"},
#     }
#
#     def __init__(self, user_input):
#         """Initialize the RK4 solver."""
#         parsed_input = self.parse_solver_input(RK4Solver.config, user_input)
#         super().__init__(f=parsed_input["f"], t_span=parsed_input["t_span"], y0=parsed_input["y0"],
#                          step_size=parsed_input["h"])
#         self.f = parsed_input["f"]
#         self.t_span = parsed_input["t_span"]  # Storing t_span
#         self.y0 = parsed_input["y0"]
#         self.h = parsed_input["h"]
#         self.validate_inputs()
#
#     def validate_inputs(self):
#         """Validate the inputs using the validation functions."""
#         self.f = validate_function_input(self.f, self.t0, self.y0)
#         self.y0 = validate_initial_conditions(self.y0)  # Validate initial conditions
#         self.h = validate_step_size(self.h)  # Validate step size
#         self.t_span = validate_time_span(self.t_span)  # Validate time span
#
#     def solve(self):
#         """
#         Solves an ODE using the 4th-order Runge-Kutta method.
#
#         Returns:
#             dict: Solution containing time steps and solution values.
#         """
#         t0, t_end = self.t_span
#         y0 = self.y0
#         h = self.step_size
#
#         logger.info(f"Solving ODE using RK4 from t={t0} to t={t_end} with step size h={h}.")
#
#         # Compute number of steps ensuring final step reaches t_end
#         num_steps = int(np.ceil((t_end - t0) / h))
#         t_values = np.linspace(t0, t0 + num_steps * h, num_steps + 1)
#
#         # Initialize solution array
#         y_values = np.zeros((len(t_values), np.size(y0)))
#         y_values[0] = y0
#
#         # RK4 iteration
#         for i in range(1, len(t_values)):
#             t = t_values[i - 1]
#             y = y_values[i - 1]
#
#             k1 = h * self.f(t, y)
#             k2 = h * self.f(t + h / 2, y + k1 / 2)
#             k3 = h * self.f(t + h / 2, y + k2 / 2)
#             k4 = h * self.f(t + h, y + k3)
#
#             y_values[i] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
#             logger.debug(f"Step {i}: t={t_values[i]:.4f}, y={y_values[i]}")
#
#         # Convert to 1D array if y0 was scalar
#         if np.isscalar(y0):
#             y_values = y_values.flatten()
#         elif y_values.ndim == 2 and y_values.shape[1] == 1:
#             y_values = y_values.reshape(-1)  # Ensure (N,1) becomes (N,)
#
#         self.t_values = t_values
#         self.y_values = y_values
#
#         logger.info("RK4 solver completed successfully.")
#         return self.get_solution()