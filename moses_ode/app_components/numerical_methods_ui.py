import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sympy as sp # Import sympy once at the top
import re # Import regex

# Import necessary functions from the library
from moses_ode.numerical.integration import trapezoid, simpson_1_3, simpson_3_8, romberg
from moses_ode.numerical.differentiation import forward_difference, backward_difference, central_difference, second_derivative

# --- LaTeX Converter for Numerical Functions ---
def convert_numerical_function_to_latex(func_string, var='x'):
    """
    Converts a single-variable function string (e.g., f(x)) to LaTeX format.
    Uses SymPy for parsing and rendering.
    """
    if not func_string or func_string.strip() == "":
        return "" # Return empty string if input is empty

    # Prepare string for SymPy parsing
    sympy_str = func_string.replace('^', '**').replace('ln(', 'log(')

    # Define the symbolic variable
    x_sym = sp.Symbol(var)

    # Define allowed functions/constants for SymPy parsing
    sympy_namespace = {
        var: x_sym,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "atan2": sp.atan2,
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "exp": sp.exp, "log": sp.log, # log is natural log in SymPy
        "log10": lambda arg: sp.log(arg, 10),
        "log2": lambda arg: sp.log(arg, 2),
        "sqrt": sp.sqrt, "cbrt": lambda arg: arg**(1/3),
        "abs": sp.Abs,
        "pi": sp.pi, "E": sp.E, "I": sp.I
    }

    try:
        # Parse the expression using SymPy
        expr = sp.sympify(sympy_str, locals=sympy_namespace)
        # Return the LaTeX representation
        return sp.latex(expr)
    except (sp.SympifyError, SyntaxError, TypeError) as parse_err:
        # If parsing fails, return the original string (or an error indicator)
        # Returning the (potentially invalid) string allows user to see what they typed
        return func_string # Fallback to the processed string
    except Exception as e:
        # Catch other errors
        # Optionally log this error
        return func_string # Fallback

# Helper function to safely parse and evaluate a function string using SymPy
def _prepare_function(func_string, var='x'):
    """
    Parses a function string f(var) using SymPy and returns a callable NumPy function.
    Consistent with SymPy parsing used elsewhere, tailored for single-variable functions.
    Handles parsing errors.
    """
    if not func_string:
        st.error("Function string is empty.")
        return lambda val: np.nan # Return a function that returns NaN

    # Prepare string for SymPy: replace ^ and handle ln
    sympy_str = func_string.replace('^', '**').replace('ln(', 'log(')

    # Define the symbolic variable
    x_sym = sp.Symbol(var)

    # Define allowed functions/constants for SymPy parsing
    # Map standard names to their SymPy equivalents
    sympy_namespace = {
        var: x_sym,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "atan2": sp.atan2,
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "exp": sp.exp, "log": sp.log, # log is natural log in SymPy
        "log10": lambda arg: sp.log(arg, 10), # Define log base 10
        "log2": lambda arg: sp.log(arg, 2),  # Define log base 2
        "sqrt": sp.sqrt, "cbrt": lambda arg: arg**(1/3), # Cube root
        "abs": sp.Abs,
        "pi": sp.pi, "E": sp.E, "I": sp.I # Constants (E is exp(1))
        # Add other common functions if needed
    }

    try:
        # Parse the expression using SymPy with the defined namespace
        expr = sp.sympify(sympy_str, locals=sympy_namespace)

        # Create a callable NumPy function using lambdify
        # Using 'numpy' and 'sympy' modules allows lambdify to translate
        # recognized SymPy functions to their NumPy equivalents.
        f_callable = sp.lambdify(x_sym, expr, modules=['numpy', 'sympy'])

        # Wrap the callable for evaluation error handling
        def safe_f_callable(val):
            try:
                result = f_callable(val)
                # Handle potential complex results if they arise and aren't desired
                if isinstance(result, complex):
                    # Check if the imaginary part is negligible
                    if np.isclose(result.imag, 0):
                        result = result.real
                    else:
                        st.warning(f"Function resulted in complex number ({result}) at {var}={val}. Returning NaN.")
                        return np.nan
                # Ensure output is float, handle potential NaNs from input domain issues
                if result is None or np.isnan(result) or np.isinf(result):
                     return np.nan
                return float(result)
            except Exception as eval_err:
                # Catch errors during the *evaluation* of the lambdified function
                # Examples: division by zero, log of negative number, etc.
                # st.warning(f"Evaluation error at {var}={val}: {eval_err}") # Optional: show eval warnings
                return np.nan # Return NaN for points where function is undefined/errors

        return safe_f_callable

    except (sp.SympifyError, SyntaxError, TypeError) as parse_err:
        # Catch errors during the *parsing* (sympify) step
        st.error(f"Error parsing function '{func_string}': {parse_err}")
        st.info(f"Check syntax (use '**' for power). Use '{var}' as the variable. Supported functions include: sin, cos, exp, log, sqrt, pi, E.")
        return lambda val: np.nan # Return NaN function
    except Exception as e:
        # Catch any other unexpected errors during preparation
        st.error(f"An unexpected error occurred preparing function: {e}")
        return lambda val: np.nan # Return NaN function


# Helper function for symbolic analysis (common pattern)
def _perform_symbolic_analysis(func_string, method_type, numerical_result, **kwargs):
    """Performs symbolic integration or differentiation for error analysis."""
    st.markdown("### Error Analysis")
    try:
        x_sym = sp.Symbol('x')
        # Prepare string for SymPy
        sympy_str = (func_string
                     .replace("^", "**")
                     .replace("ln(", "log(")) # SymPy uses log for natural log
        
        # Define namespace for sympify
        sympy_namespace = {
            "sp": sp, "x": x_sym,
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
            "pi": sp.pi, "E": sp.E, "I": sp.I # Added E and I just in case
        }
        
        try:
            expr = sp.sympify(sympy_str, locals=sympy_namespace)
        except (sp.SympifyError, SyntaxError, TypeError) as sym_err:
            st.warning(f"SymPy parsing failed: {sym_err}. Cannot perform error analysis.")
            st.info("Ensure function uses standard SymPy-compatible syntax (e.g., x**2, sp.sin(x)).")
            return

        if method_type == 'integration':
            a_val = kwargs['a_value']
            b_val = kwargs['b_value']
            integration_method = kwargs['integration_method']
            f_callable = kwargs['f_callable'] # Need the callable function too

            # Symbolic integration
            integral_expr = sp.integrate(expr, x_sym)
            st.markdown("Symbolic antiderivative:")
            st.latex(sp.latex(integral_expr))

            try:
                # Evaluate definite integral
                exact_result = float(integral_expr.subs(x_sym, b_val) - integral_expr.subs(x_sym, a_val))
                error = abs(numerical_result - exact_result)
                relative_error = error / abs(exact_result) if exact_result != 0 else float('inf')

                st.write(f"Numerical approximation: {numerical_result:.10f}")
                st.write(f"Exact integral value: {exact_result:.10f}")
                st.write(f"Absolute error: {error:.12f}")
                st.write(f"Relative error: {relative_error:.12f}")

                # Error convergence plot (if applicable)
                # ... (Code for convergence plot as in the original app) ...
                # This part needs the numerical methods (trapezoid, simpson etc.) and f_callable
                # Simplified for brevity, but should be included from original logic

            except (TypeError, ValueError) as eval_ex:
                st.warning(f"Could not evaluate the exact integral symbolically: {str(eval_ex)}")

        elif method_type == 'differentiation':
            x0_val = kwargs['x_value']
            diff_method = kwargs['differentiation_method']
            f_callable = kwargs['f_callable'] # Need the callable function
            h_val = kwargs['h_value'] # Need step size

            # Symbolic differentiation
            derivative_expr = sp.diff(expr, x_sym, 1 if diff_method != "Second Derivative" else 2)
            st.markdown("Symbolic derivative:")
            st.latex(sp.latex(derivative_expr))

            try:
                # Evaluate exact derivative
                derivative_func = sp.lambdify(x_sym, derivative_expr, modules=['numpy', 'sympy'])
                exact_result = float(derivative_func(x0_val))
                error = abs(numerical_result - exact_result)
                relative_error = error / abs(exact_result) if exact_result != 0 else float('inf')

                st.write(f"Numerical approximation: {numerical_result:.10f}")
                st.write(f"Exact derivative value: {exact_result:.10f}")
                st.write(f"Absolute error: {error:.12f}")
                st.write(f"Relative error: {relative_error:.12f}")

                # Error convergence plot (if applicable)
                # ... (Code for convergence plot as in the original app) ...
                # This part needs numerical diff methods and f_callable

            except (TypeError, ValueError, NameError) as eval_ex:
                st.warning(f"Could not evaluate the exact derivative symbolically: {str(eval_ex)}")

    except ImportError:
        st.warning("SymPy library not found. Cannot perform symbolic error analysis. Install with: pip install sympy")
    except Exception as ex:
        st.error(f"An unexpected error occurred during symbolic analysis: {str(ex)}")


def create_integration_tab():
    """Creates the UI and logic for the Integration tab."""
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("### Integration Setup")
        integration_method = st.selectbox(
            "Select integration method:",
            ["Trapezoidal Rule", "Simpson's 1/3 Rule", "Simpson's 3/8 Rule", "Romberg Integration"],
            key="integration_method"
        )
        
        # --- Add LaTeX Preview for Integration Function ---
        # Initialize state if key doesn't exist (st.text_input does this too, but explicit is safe)
        if "integration_function" not in st.session_state:
            st.session_state.integration_function = "x^2" # Default value from input below
            
        # Display LaTeX preview if function string is not empty
        if st.session_state.integration_function:
            latex_formula = convert_numerical_function_to_latex(st.session_state.integration_function, var='x')
            st.markdown('<div class="latex-preview">', unsafe_allow_html=True)
            st.latex(f"f(x) = {latex_formula}")
            st.markdown('</div>', unsafe_allow_html=True)
        # --- End LaTeX Preview ---
            
        integration_function_value = st.text_area(
            "Function to integrate f(x)", 
            help="Enter a function of x (e.g., x^2, sin(x), exp(-x)). Press Ctrl+Enter to apply.",
            key="integration_function", # Use key for state management
            height=68 # Set a reasonable height for potentially single-line input
        )
        a_value = st.number_input("Lower bound (a)", value=0.0, step=0.1, format="%.2f", key="integration_a")
        b_value = st.number_input("Upper bound (b)", value=1.0, step=0.1, format="%.2f", key="integration_b")

        n_intervals, max_iterations, tol = None, None, None
        if integration_method != "Romberg Integration":
            n_intervals = st.slider("Number of intervals", min_value=10, max_value=1000, value=100, step=10, key="integration_n")
        else:
            max_iterations = st.slider("Maximum iterations", min_value=2, max_value=10, value=6, step=1, key="romberg_max_iter")
            tol = st.number_input("Tolerance", value=1e-10, format="%.2e", key="romberg_tol")

        # Add checkbox for symbolic analysis
        show_symbolic = st.checkbox("Show Symbolic Error Analysis", key="integration_show_symbolic")
        
        compute_integral = st.button("Compute Integral", key="compute_integral_btn", use_container_width=True)

    with col2:
        st.markdown("### Results")
        if compute_integral:
            if not st.session_state.integration_function:
                st.error("Please enter a function to integrate.")
                return
            
            # Prepare the function using the value from session state
            f_callable = _prepare_function(st.session_state.integration_function, var='x')
            
            # Check if function preparation failed (indicated by returning None or similar)
            # For now, assume it works or _prepare_function shows error and returns NaN-producing func
            
            start_time = time.time()
            result, method_details, R_table = None, "", None

            try:
                if integration_method == "Trapezoidal Rule":
                    n = n_intervals
                    result = trapezoid(f_callable, a_value, b_value, n)
                    method_details = f"Number of intervals: {n}"
                elif integration_method == "Simpson's 1/3 Rule":
                    n = n_intervals
                    if n % 2 != 0: n += 1
                    result = simpson_1_3(f_callable, a_value, b_value, n)
                    method_details = f"Number of intervals: {n}"
                elif integration_method == "Simpson's 3/8 Rule":
                    n = n_intervals
                    if n % 3 != 0: n += 3 - (n % 3)
                    result = simpson_3_8(f_callable, a_value, b_value, n)
                    method_details = f"Number of intervals: {n}"
                else: # Romberg
                    max_iter = max_iterations
                    result, R_table = romberg(f_callable, a_value, b_value, max_iter, tol)
                    method_details = f"Max iterations: {max_iter}, Tolerance: {tol:.2e}"
                    if not isinstance(R_table, np.ndarray): R_table = np.array([[result]]) # Ensure table exists

                compute_time = time.time() - start_time

                st.success(f"Integral Result: {result:.10f}")
                st.info(f"Method: {integration_method}\n{method_details}\nComputation Time: {compute_time:.6f} seconds")

                # Plotting (reuse original plotting logic)
                fig, ax = plt.subplots(figsize=(10, 6))
                x_points = np.linspace(a_value, b_value, 500) # Reduced points for plot
                y_points = np.array([f_callable(x) for x in x_points])
                
                # Filter NaNs for plotting
                valid_mask = ~np.isnan(y_points)
                if np.any(valid_mask):
                     ax.plot(x_points[valid_mask], y_points[valid_mask], 'b-', label=f'f(x) = {st.session_state.integration_function}')
                     ax.fill_between(x_points[valid_mask], y_points[valid_mask], alpha=0.2, color='b')
                else:
                    st.warning("Could not plot function (all points resulted in errors or NaN).")

                ax.set_xlabel('x'); ax.set_ylabel('f(x)')
                ax.set_title(f'Numerical Integration: {integration_method}')
                ax.grid(True, linestyle='--', alpha=0.7); ax.legend()
                st.pyplot(fig); plt.close(fig)

                # Romberg Table
                if R_table is not None:
                    st.markdown("### Romberg Integration Table")
                    try:
                        n_rows, n_cols = R_table.shape
                        if n_rows > 0 and n_cols > 0:
                            romberg_df = pd.DataFrame(R_table, columns=[f"R(i,{j})" for j in range(n_cols)])
                            romberg_df.insert(0, "Iteration", [f"i={i}" for i in range(n_rows)])
                            st.dataframe(romberg_df)
                        else:
                            st.warning("Romberg table is empty.")
                    except Exception as e:
                        st.warning(f"Error displaying Romberg table: {e}")

                # Conditionally perform symbolic analysis
                if show_symbolic:
                     _perform_symbolic_analysis(
                         st.session_state.integration_function, 
                         'integration', 
                         result, 
                         a_value=a_value, 
                         b_value=b_value,
                         integration_method=integration_method,
                         f_callable=f_callable
                         # Add other necessary kwargs if needed by the plot part
                     )

            except Exception as e:
                st.error(f"Error computing integral: {str(e)}")


def create_differentiation_tab():
    """Creates the UI and logic for the Differentiation tab."""
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("### Differentiation Setup")
        differentiation_method = st.selectbox(
            "Select differentiation method:",
            ["Forward Difference", "Backward Difference", "Central Difference", "Second Derivative"],
            key="differentiation_method"
        )
        
        # --- Add LaTeX Preview for Differentiation Function ---
        # Initialize state if key doesn't exist
        if "differentiation_function" not in st.session_state:
            st.session_state.differentiation_function = "sin(x)" # Default value from input below
            
        # Display LaTeX preview if function string is not empty
        if st.session_state.differentiation_function:
            latex_formula = convert_numerical_function_to_latex(st.session_state.differentiation_function, var='x')
            st.markdown('<div class="latex-preview">', unsafe_allow_html=True)
            st.latex(f"f(x) = {latex_formula}")
            st.markdown('</div>', unsafe_allow_html=True)
        # --- End LaTeX Preview ---
            
        differentiation_function_value = st.text_area(
            "Function to differentiate f(x)", 
            help="Enter a function of x (e.g., x^2, sin(x), exp(x)). Press Ctrl+Enter to apply.",
            key="differentiation_function", # Use key for state management
            height=68 # Set a reasonable height
        )
        x_value = st.number_input("Point to evaluate at (x)", value=1.0, step=0.1, format="%.4f", key="differentiation_x")
        h_value = st.number_input("Step size (h)", value=1e-5, min_value=1e-10, max_value=0.1, format="%.2e", key="differentiation_h")

        # Add checkbox for symbolic analysis
        show_symbolic = st.checkbox("Show Symbolic Error Analysis", key="diff_show_symbolic")

        compute_derivative = st.button("Compute Derivative", key="compute_derivative_btn", use_container_width=True)

    with col2:
        st.markdown("### Results")
        if compute_derivative:
            if not st.session_state.differentiation_function:
                st.error("Please enter a function to differentiate.")
                return

            f_callable = _prepare_function(st.session_state.differentiation_function, var='x')

            start_time = time.time()
            result, formula = None, ""
            x0, h = x_value, h_value
            method = differentiation_method
            derivative_order = "first" if method != "Second Derivative" else "second"

            try:
                if method == "Forward Difference":
                    result = forward_difference(f_callable, x0, h)
                    formula = f"f'(x) ≈ [f(x+h) - f(x)] / h"
                elif method == "Backward Difference":
                    result = backward_difference(f_callable, x0, h)
                    formula = f"f'(x) ≈ [f(x) - f(x-h)] / h"
                elif method == "Central Difference":
                    result = central_difference(f_callable, x0, h)
                    formula = f"f'(x) ≈ [f(x+h) - f(x-h)] / (2h)"
                else: # Second Derivative
                    result = second_derivative(f_callable, x0, h)
                    formula = f"f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²"

                compute_time = time.time() - start_time
                
                derivative_symbol = "f'(x)" if derivative_order == "first" else "f''(x)"
                st.success(f"{derivative_symbol} at x = {x0}: {result:.10f}")
                st.info(f"Method: {method}\nFormula: {formula}\nStep size (h): {h}\nComputation Time: {compute_time:.6f} seconds")

                # Plotting (reuse original logic)
                fig, ax = plt.subplots(figsize=(10, 6))
                x_range = max(2*h, abs(x0) * 0.5) + h # Ensure range includes points needed for diff viz
                x_points = np.linspace(x0 - x_range, x0 + x_range, 500)
                y_points = np.array([f_callable(x_i) for x_i in x_points])
                
                valid_mask = ~np.isnan(y_points)
                if np.any(valid_mask):
                    ax.plot(x_points[valid_mask], y_points[valid_mask], 'b-', label=f'f(x) = {st.session_state.differentiation_function}')
                    
                    # Plot tangent/secant lines (check points exist and are valid)
                    try:
                        fx0 = f_callable(x0)
                        if not np.isnan(fx0):
                            ax.plot(x0, fx0, 'ro', markersize=8, label=f'Point (x={x0})')
                            if derivative_order == "first":
                                tangent_points = result * (x_points - x0) + fx0
                                ax.plot(x_points[valid_mask], tangent_points[valid_mask], 'r--', label=f'Approx. Tangent (Slope={result:.3f})')
                            
                            # Visualization points
                            if method == "Forward Difference" and not np.isnan(f_callable(x0+h)):
                                ax.plot([x0, x0+h], [fx0, f_callable(x0+h)], 'g-o', markersize=5, label='Forward Diff Pts')
                            elif method == "Backward Difference" and not np.isnan(f_callable(x0-h)):
                                ax.plot([x0-h, x0], [f_callable(x0-h), fx0], 'g-o', markersize=5, label='Backward Diff Pts')
                            elif method == "Central Difference" and not np.isnan(f_callable(x0-h)) and not np.isnan(f_callable(x0+h)):
                                ax.plot([x0-h, x0+h], [f_callable(x0-h), f_callable(x0+h)], 'g-o', markersize=5, label='Central Diff Pts')
                        else:
                             st.warning(f"Cannot plot point or tangent: f({x0}) resulted in NaN.")

                    except Exception as plot_detail_err:
                         st.warning(f"Error plotting details: {plot_detail_err}")
                else:
                    st.warning("Could not plot function (all points resulted in errors or NaN).")

                ax.set_xlabel('x'); ax.set_ylabel('f(x)')
                ax.set_title(f'Numerical Differentiation: {method}')
                ax.grid(True, linestyle='--', alpha=0.7); ax.legend(fontsize='small')
                st.pyplot(fig); plt.close(fig)

                # Conditionally perform symbolic analysis
                if show_symbolic:
                    _perform_symbolic_analysis(
                        st.session_state.differentiation_function, 
                        'differentiation', 
                        result, 
                        x_value=x_value,
                        differentiation_method=method,
                        f_callable=f_callable,
                        h_value=h_value
                    )

            except Exception as e:
                st.error(f"Error computing derivative: {str(e)}")


def create_numerical_methods_tab():
    """Creates the main container for the Numerical Methods tab."""
    
    # --- Add CSS for LaTeX Preview (Moved Here) ---
    st.markdown("""
    <style>
    .latex-preview {
        background-color: #f0f8ff;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 10px; /* Space below preview, above input */
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)
    # --- End CSS ---
    
    st.markdown("## Numerical Methods")
    integration_tab, differentiation_tab = st.tabs(["Integration", "Differentiation"])

    with integration_tab:
        create_integration_tab()

    with differentiation_tab:
        create_differentiation_tab() 