"""
Defines standard ODE problems for benchmarking solvers.
"""

import numpy as np

# Problem definitions structure:
# 'name': {
#     'ode_str': String representation of the ODE function (or system).
#     'y0_str': String representation of the initial condition(s).
#     't_span': Tuple (t0, t_end) for the integration interval.
#     'analytical_sol': Optional function `f(t)` for the exact solution.
# }

def decay_analytical(t):
    return np.exp(-2 * t)

def oscillatory_analytical_y0(t): # y0 = [0, 1] -> y(t) = sin(t)
      return np.sin(t)

# Note: Stiff problem's analytical solution is more complex and omitted here
# for simplicity in error calculation within the benchmark CLI command.
# y(t) = [exp(-t), exp(-1000*t)] for y0=[1,1]
# For y0=[1,0], it's y(t) = [ (1000/999)*exp(-t) - (1/999)*exp(-1000*t),
#                           (-1000/999)*exp(-t) + (1000/999)*exp(-1000*t) ]
# We will only benchmark execution, not accuracy vs analytical for stiff.

PROBLEMS = {
    "stiff": {
        "ode_str": "[y[1], -1000*y[0] - 1001*y[1]]", # Note: Using y[0], y[1] indexing
        "y0_str": "[1.0, 0.0]",
        "t_span": (0.0, 10.0),
        "analytical_sol": None # Too complex for simple benchmark error check
    },
    "oscillatory": {
        "ode_str": "[y[1], -y[0]]", # Note: Using y[0], y[1] indexing
        "y0_str": "[0.0, 1.0]",
        "t_span": (0.0, 10.0),
        "analytical_sol": oscillatory_analytical_y0 # Only check error for y[0] = sin(t)
    },
    "decay": {
        "ode_str": "-2*y",
        "y0_str": "1.0",
        "t_span": (0.0, 5.0),
        "analytical_sol": decay_analytical
    }
}

def get_problem(name: str):
    """Retrieve a benchmark problem by name."""
    problem = PROBLEMS.get(name)
    if problem is None:
        raise ValueError(f"Unknown benchmark problem: {name}")
    # Potentially add validation here if needed
    return problem 