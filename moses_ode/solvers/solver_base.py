from abc import ABC, abstractmethod
import numpy as np
import logging
from scipy.interpolate import interp1d


class ODESolver(ABC):
    """
    Abstract base class for all ODE solvers.
    Defines a common structure that all solvers must follow.
    """

    def __init__(self, f, t_span, y0, step_size=None, tolerances=None, events=None, log_level=logging.INFO):
        """
        Initialize the solver.

        Parameters:
        - f: Callable, the ODE function f(t, y).
        - t_span: Tuple (t0, tf), the start and end time.
        - y0: Initial condition (scalar or array).
        - step_size: Optional step size for fixed-step methods.
        - tolerances: Tuple (rtol, atol) for adaptive methods.
        - events: List of event functions to terminate integration.
        - log_level: Logging level (e.g., logging.INFO, logging.DEBUG).
        """
        # Input validation - call the methods
        if not callable(f):
            raise TypeError("f must be a callable function.")
        if not isinstance(t_span, (tuple, list)) or len(t_span) != 2:
            raise ValueError("t_span must be a tuple or list of length 2.")
        self.t0, self.tf = t_span
        if self.t0 >= self.tf:
            raise ValueError("t0 must be less than tf.")

        self.t_span = t_span  # Explicitly storing t_span
        self.y0 = np.array(y0, dtype=float)
        self.step_size = step_size if step_size else (self.tf - self.t0) / 100  # Default: 100 steps
        self.tolerances = tolerances
        self.events = events

        # Initialize solution storage
        self.solution = None
        self._initialize_solution()

        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        self.logger.info(f"Initializing {self.__class__.__name__} solver...")
        self.logger.info(f"t_span: {self.t_span}, step_size: {self.step_size}")

    @abstractmethod
    def solve(self):
        """
        Solve the ODE. This must be implemented by subclasses.
        """
        pass

    def _initialize_solution(self):
        """
        Helper method to set up storage for solution.
        """
        self.t_values = [self.t0]
        self.y_values = [self.y0]

    def _store_step(self, t, y):
        """
        Store a computed step in the solution arrays.
        """
        self.t_values.append(t)
        self.y_values.append(y)
        self.logger.debug(f"Step stored: t={t}, y={y}")

    def _check_events(self, t, y):
        """
        Check if any event has occurred.
        """
        if self.events:
            for event in self.events:
                if event(t, y) == 0:
                    return True
        return False

    def dense_output(self, t):
        """
        Interpolate the solution at time t.
        """
        t_values, y_values = self.get_solution().values()
        return interp1d(t_values, y_values, axis=0)(t)

    def get_solution(self):
        """
        Return the computed solution as a dictionary.
        """
        return {
            "t": np.array(self.t_values),
            "y": np.array(self.y_values)
        }

    def __repr__(self):
        """
        String representation of the solver.
        """
        return f"{self.__class__.__name__}(t_span={self.t_span}, step_size={self.step_size})"

    def parse_solver_input(self, config, user_input):
        """Parses solver input based on a configuration dictionary."""
        parsed_input = {}
        for param, spec in config.items():
            if spec["required"] and param not in user_input:
                raise ValueError(f"Missing required parameter: {param}")

            if param in user_input:
                value = user_input[param]
                if not isinstance(value, spec["type"]):
                    raise TypeError(
                        f"Parameter {param} must be of type {spec['type']}, got {type(value)}")
                parsed_input[param] = value
            elif "default" in spec:
                parsed_input[param] = spec["default"]
        return parsed_input