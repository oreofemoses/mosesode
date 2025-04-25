# File: moses_ode/parsing/symbolic_parser.py
import sympy as sp
import logging
import re # Import the regex module

logger = logging.getLogger(__name__)

def convert_to_first_order(ode, y, t):
    """
    Converts a nth-order ODE into a system of first-order ODEs.

    Parameters:
    - ode: A symbolic equation (e.g., d²y/dt² = -9.81).
    - y: The dependent variable (e.g., y(t)).
    - t: The independent variable (time t).

    Returns:
    - system: List of first-order ODEs.
    - vars: List of dependent variables [y1, y2, ...].

    Example:
    >>> t = sp.symbols('t')
    >>> y = sp.Function('y')(t)
    >>> ode = sp.Eq(sp.diff(y, t, t), -9.81)
    >>> system, vars = convert_to_first_order(ode, y, t)
    >>> system
    [Eq(Derivative(y1(t), t), y2(t)), Eq(Derivative(y2(t), t), -9.81)]
    >>> vars
    [y1(t), y2(t)]
    """
    # Input validation
    if not isinstance(ode, sp.Eq):
        raise TypeError("ode must be a SymPy equation.")
    if not isinstance(y, sp.Function):
        raise TypeError("y must be a SymPy function.")
    if not isinstance(t, sp.Symbol):
        raise TypeError("t must be a SymPy symbol.")

    logger.info("Converting ODE to first-order system...")

    # Find the highest derivative
    derivatives = list(ode.atoms(sp.Derivative))
    if not derivatives:
        raise ValueError("No derivatives found in the equation. Ensure the ODE is expressed in terms of derivatives.")

    highest_derivative = max(derivatives, key=lambda d: d.derivative_count)
    order = highest_derivative.derivative_count  # Order of the ODE

    # If already first order, return as-is
    if order == 1:
        logger.info("ODE is already first order.")
        return [ode], [y]

    # Define new variables for each derivative
    vars = [sp.Function(f'y{i + 1}')(t) for i in range(order)]

    # Solve for the highest derivative explicitly
    try:
        highest_derivative_expr = sp.solve(ode, highest_derivative)
        if not highest_derivative_expr:
            raise ValueError("Could not solve for highest derivative.")
        highest_derivative_expr = highest_derivative_expr[0]  # Take the first solution if multiple
    except Exception as e:
        raise ValueError(f"Error solving for highest derivative: {e}")

    # Create the system of first-order ODEs
    system = [sp.Eq(sp.Derivative(vars[i], t), vars[i + 1]) for i in range(order - 1)]
    system.append(sp.Eq(sp.Derivative(vars[-1], t), highest_derivative_expr))

    logger.info("Conversion complete.")
    return system, vars

def sys_of_ode (ODE):
    """
    Converts a higher-order ODE in D-notation to a system of first-order ODEs
    represented by the list of derivatives [y1, y2, ..., D^n y].

    Parameters:
    - ODE: String representing the ODE in D-notation (e.g., "D2y = -9.81", "Dy = -y").

    Returns:
    - String representing the right-hand side of the system in the format "[expr1, expr2, ..., expr_n]".
      For an n-th order ODE, this will be "[y1, y2, ..., final_expression]".
      For a 1st order ODE, this will be "[final_expression]".

    Example:
    >>> sys_of_ode("D2y = -9.81")
    "[y1, -9.81]"
    >>> sys_of_ode("D3y = -D2y - D1y - y")
    "[y1, y2, -y2 - y1 - y0]"
    >>> sys_of_ode("Dy = -k*y")
    "[-k*y0]"
    """
    # Prepare the ODE string
    ode_str = ODE.strip()
    lhs, rhs = ode_str.split('=', 1)
    lhs = lhs.strip()
    system_rhs = rhs.strip() # Store the original RHS for later processing

    # Determine the order
    if lhs == 'Dy':
        order = 1
    elif lhs.startswith('D') and lhs.endswith('y'):
        try:
            order = int(lhs[1:-1])
        except ValueError:
            raise ValueError(f"Cannot parse order from LHS: {lhs}")
    else:
        raise ValueError(f"Invalid LHS format for D-notation: {lhs}. Expected 'Dy' or 'D<n>y'.")

    # --- Handle First Order Case ---
    if order == 1:
        # Replace standalone 'y' with 'y0' using regex for whole word match
        processed_rhs = re.sub(r'\by\b', 'y0', system_rhs) 
        # Replace 'Dy' or 'D1y' if they somehow appear (shouldn't in valid RHS)
        processed_rhs = processed_rhs.replace('D1y', 'y1').replace('Dy', 'y1')
        return f'[{processed_rhs.strip()}]'

    # --- Handle Higher Order Cases (order > 1) ---
    
    # Build the list of derivatives [y1, y2, ..., y(n-1)]
    result_elements = [f"y{i}" for i in range(1, order)]

    # Prepare the final expression (RHS of the highest derivative)
    # Replace Dy with D1y first for consistency
    processed_rhs = system_rhs.replace('Dy', 'D1y')
    
    # IMPORTANT: Replace standalone 'y' with 'y0' using regex BEFORE replacing derivatives
    processed_rhs = re.sub(r'\by\b', 'y0', processed_rhs)
    
    # Replace D{k}y with y{k} (k = order-1 down to 1)
    # Iterate downwards to avoid partial replacements (e.g. D1y in D10y)
    for k in range(order - 1, 0, -1):
        processed_rhs = processed_rhs.replace(f'D{k}y', f'y{k}')
    
    # Add the processed final expression to the list
    result_elements.append(processed_rhs.strip())

    # Join the elements into the final string format
    return f'[{", ".join(result_elements)}]'

