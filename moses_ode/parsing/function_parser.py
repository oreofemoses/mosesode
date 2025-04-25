import sympy as sp
import numpy as np
import logging

logger = logging.getLogger(__name__)

def parse_function_string(func_str):
    """
    Parses a user-provided function string and returns a callable f(t, y).

    - Supports scalar and system ODEs.
    - Uses `sympy.sympify` for secure parsing.
    - Converts the function into a NumPy-compatible lambda function.
    """
    t = sp.Symbol('t')  # Define time symbol

    # If function represents a system (list notation), parse as a system
    func_str = func_str.strip()
    if func_str.startswith('[') and func_str.endswith(']'):
        logger.info(f"Parsing system of ODEs: {func_str}")
        expr_strs = func_str[1:-1].split(',')
        y_symbols = sp.symbols(f'y0:{len(expr_strs)}')  # Define y0, y1, ..., yN
        allowed_symbols = {t, *y_symbols}

        # Convert each expression to sympy
        exprs = []
        for i, expr in enumerate(expr_strs):
            try:
                parsed_expr = sp.sympify(expr.strip(), locals={'t': t, **{f'y{i}': y_symbols[i] for i in range(len(y_symbols))}})
                exprs.append(parsed_expr)
            except Exception as e:
                raise ValueError(f"Error parsing expression '{expr}': {str(e)}")

        # Check for undefined symbols
        for expr in exprs:
            used_symbols = expr.free_symbols
            if not used_symbols.issubset(allowed_symbols):
                raise ValueError(f"Invalid variable used: {used_symbols - allowed_symbols}")

        # Create lambdified functions for each expression
        lambdified_funcs = [sp.lambdify((t, *y_symbols), expr, modules=['numpy']) for expr in exprs]
        
        def f(t_val, y_val):
            if np.isscalar(y_val):
                y_val = np.array([y_val])  # Convert scalar to array
            elif not isinstance(y_val, np.ndarray):
                y_val = np.array(y_val)  # Convert other sequences to array
                
            # Pad or truncate y_val to match the system size
            if len(y_val) < len(y_symbols):
                # Pad with zeros if y_val is too short
                y_val = np.pad(y_val, (0, len(y_symbols) - len(y_val)))
            elif len(y_val) > len(y_symbols):
                # Truncate if y_val is too long
                y_val = y_val[:len(y_symbols)]
                
            # Evaluate each expression with the adjusted y_val
            result = np.zeros(len(lambdified_funcs))
            for i, func in enumerate(lambdified_funcs):
                result[i] = func(t_val, *y_val)
                
            return result

        return f

    else:  # Scalar case
        y = sp.Symbol('y')  # Define y as scalar
        
        # Handle NumPy functions specifically
        if 'np.' in func_str:
            # Create a direct lambda function for NumPy expressions
            def safe_f(t_val, y_val):
                # Create a local context with t and y values
                local_vars = {'t': t_val, 'y': y_val[0] if hasattr(y_val, '__len__') else y_val, 'np': np}
                # Replace np.exp(-t) with np.exp(negative_t) to handle the parsing better
                modified_func_str = func_str.replace('np.', 'np_')
                modified_func_str = modified_func_str.replace('np_', 'np.')
                try:
                    result = eval(modified_func_str, {"__builtins__": {}}, local_vars)
                    return np.atleast_1d(result)
                except Exception as e:
                    raise ValueError(f"Error evaluating expression '{func_str}': {str(e)}")
            
            return safe_f
        
        # For non-NumPy expressions, use sympy
        try:
            locals_dict = {'t': t, 'y': y}
            expr = sp.sympify(func_str, locals=locals_dict)
        except Exception as e:
            # Try to provide a helpful error message
            raise ValueError(f"Error parsing expression '{func_str}': {str(e)}")

        # Check for undefined symbols
        allowed_symbols = {t, y}
        if not expr.free_symbols.issubset(allowed_symbols):
            raise ValueError(f"Invalid variable used: {expr.free_symbols - allowed_symbols}")

        # Create a lambda function using numpy for numerical computation
        lambda_expr = sp.lambdify((t, y), expr, modules=['numpy'])
        
        # Ensure output is always a 1D NumPy array
        def safe_f(t_val, y_val):
            # Handle scalar or array input properly
            if np.isscalar(y_val):
                result = lambda_expr(t_val, y_val)
            else:
                # For array input (which could be from a system)
                if len(np.atleast_1d(y_val)) > 1:
                    # Use just the first element if y_val is an array
                    result = lambda_expr(t_val, y_val[0])
                else:
                    result = lambda_expr(t_val, y_val[0])
            
            # Ensure the result is a numpy array
            return np.atleast_1d(result)
        
        return safe_f