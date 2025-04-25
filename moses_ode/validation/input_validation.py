import numpy as np
import logging

logger = logging.getLogger(__name__)


def validate_function_input(f, t0, y0, *args):
    """
    Validates that the user-defined ODE function f(t, y, *args) is correctly formatted.
    Ensures:
    - f is callable
    - f(t, y) returns a NumPy array of the same shape as y0
    - f does NOT return a list (to prevent unintended shape conversions)
    """
    if not callable(f):
        raise TypeError("ODE function f must be callable: f(t, y).")

    y0 = np.atleast_1d(np.asarray(y0, dtype=float))  # Ensures at least 1D array

    try:
        # First try with potential scalar input
        if y0.size == 1:
            # Try scalar first (some functions might expect a scalar for simple ODEs)
            try:
                dydt_scalar = f(t0, float(y0[0]), *args)
                # If it worked, convert to array format
                dydt = np.atleast_1d(dydt_scalar)
            except Exception:
                # If scalar fails, try with array
                dydt = f(t0, y0, *args)
        else:
            # For vector input, use normal approach
            dydt = f(t0, y0, *args)
            
        if isinstance(dydt, list):
            dydt = np.array(dydt)  # Convert lists to numpy arrays
            
        dydt = np.atleast_1d(np.asarray(dydt, dtype=float))
        
    except TypeError as te: # Catch TypeError specifically
        raise TypeError(f"Function evaluation failed: {str(te)}")
    except Exception as e: # Catch other exceptions as ValueErrors
        raise ValueError(f"Error when calling f(t, y): {str(e)}")

    logger.debug(f"dydt={dydt}, dydt.shape={dydt.shape}, expected shape={y0.shape}")
    
    # --- Add Debugging --- 
    # print(f"DEBUG: validate_function_input: y0.shape={y0.shape}, y0.size={y0.size}")
    # print(f"DEBUG: validate_function_input: dydt.shape={dydt.shape}, dydt.size={dydt.size}")
    # --- End Debugging ---

    # For scalar ODEs, we're more flexible about the shape
    if y0.size == 1 and dydt.size == 1:
        # Both are scalar-like, so we're good
        pass
    elif dydt.shape != y0.shape:
        # Try to reshape if possible
        if dydt.size == y0.size:
            try:
                dydt = dydt.reshape(y0.shape)
            except:
                # If reshape fails, it's truly an incompatible shape
                error_msg = f"f(t, y) must return the same shape as y0. Got {dydt.shape}, expected {y0.shape}."
                # Add helpful message for higher-order ODEs
                if dydt.size > y0.size:
                    # Special case for systems generated from higher-order ODEs
                    # Check if this might be a system representation of a higher-order ODE
                    # For an n-th order ODE, the system has n+1 elements but needs n initial conditions
                    if dydt.size == y0.size + 1:
                        num_needed = y0.size
                        error_msg = f"For a {num_needed}-order ODE, you need to provide exactly {num_needed} initial conditions: [y(t0), y'(t0), ..., y^({num_needed-1})(t0)]"
                    else:
                        # Standard case
                        num_needed = dydt.size
                        error_msg += f"\nFor a {num_needed-1}-order ODE, you need to provide exactly {num_needed} initial conditions: [y(t0), y'(t0), ..., y^({num_needed-2})(t0)]"
                raise ValueError(error_msg)
        else:
            error_msg = f"f(t, y) must return the same shape as y0. Got {dydt.shape}, expected {y0.shape}."
            if dydt.size > y0.size:
                # Special case for systems generated from higher-order ODEs
                if dydt.size == y0.size + 1:
                    num_needed = y0.size
                    error_msg = f"For a {num_needed}-order ODE, you need to provide exactly {num_needed} initial conditions: [y(t0), y'(t0), ..., y^({num_needed-1})(t0)]"
                else:
                    num_needed = dydt.size
                    error_msg += f"\nFor a {num_needed-1}-order ODE, you need to provide exactly {num_needed} initial conditions: [y(t0), y'(t0), ..., y^({num_needed-2})(t0)]"
            raise ValueError(error_msg)

    logger.info("Function input validated successfully.")
    return f

def validate_jacobian_input(jac, t0, y0, *args):
    """
    Validates that the user-defined Jacobian function jac(t, y, *args) is correctly formatted.
    Ensures:
    - jac is callable
    - jac(t, y) returns a NumPy square matrix of shape (len(y0), len(y0))
    - jac does NOT return a list (to prevent unintended shape conversions)
    """
    if not callable(jac):
        raise TypeError("Jacobian function jac must be callable: jac(t, y).")

    y0 = np.atleast_1d(np.asarray(y0, dtype=float))  # Ensures at least 1D array
    expected_shape = (y0.size, y0.size)

    try:
        J = jac(t0, y0, *args)
        if isinstance(J, list):
            raise TypeError("Jacobian function jac must return a NumPy array, not a list.")

        J = np.asarray(J, dtype=float)
    except Exception as e:
        raise ValueError(f"Error when calling jac(t, y): {e}")

    if J.shape != expected_shape:
        raise ValueError(f"jac(t, y) must return a square matrix of shape {expected_shape}. Got {J.shape}.")

    logger.info("Jacobian input validated successfully.")
    return jac
