import streamlit as st
# Import constants needed for the sidebar (e.g., example lists)
from moses_ode.app_components.constants import EXAMPLE_ODES, ODE_LIBRARY
from moses_ode.parsing.function_parser import parse_function_string
from moses_ode.validation.input_validation import validate_function_input
import numpy as np
import time
import logging
import re
import sympy as sp

logger = logging.getLogger(__name__)

def convert_to_latex(function_str):
    """
    Converts a function string to LaTeX format for display.
    Handles special cases like 'Dny' to 'd^n y / dt^n'.
    Replaces '^' with '**' for consistent parsing.
    
    Args:
        function_str: The function string to convert
        
    Returns:
        LaTeX formatted string
    """
    # Handle empty strings
    if not function_str or function_str.strip() == "":
        return ""
    
    # Replace ^ with ** for consistent parsing before any processing
    safe_function_str = function_str.replace('^', '**')

    # Handle system of ODEs
    if safe_function_str.strip().startswith('[') and safe_function_str.strip().endswith(']'):
        # Process each equation in the system
        equations = safe_function_str.strip()[1:-1].split(',')
        latex_equations = []
        
        for i, eq in enumerate(equations):
            processed_eq = eq.strip()
            # Replace y0, y1, etc. with proper LaTeX variables
            for j in range(10):  # Assume up to y9
                processed_eq = re.sub(f'\\by{j}\\b', f'y_{{{j}}}', processed_eq)
            
            # Try to convert to LaTeX using SymPy
            try:
                expr = sp.sympify(processed_eq) # Parse the already replaced string
                latex_eq = sp.latex(expr)
                latex_equations.append(latex_eq)
            except:
                latex_equations.append(processed_eq)
        
        # Combine the system equations
        combined_latex = r'\begin{cases} '
        for i, latex_eq in enumerate(latex_equations):
            if i < len(latex_equations) - 1:
                combined_latex += f'y_{i}\'(t) = {latex_eq} \\\\ '
            else:
                combined_latex += f'y_{i}\'(t) = {latex_eq}'
        combined_latex += r' \end{cases}'
        return combined_latex

    # Check for equation with = sign
    if '=' in safe_function_str:
        lhs, rhs = safe_function_str.split('=', 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        
        # Check for D-notation (on the original LHS structure, before ** replacement)
        # Need to check original string's structure for D-notation detection
        original_lhs = function_str.split('=', 1)[0].strip() 
        if original_lhs.startswith('D') and 'y' in original_lhs:
            try:
                # Use original_lhs for parsing order
                if original_lhs == 'Dy':
                    order = 1
                else:
                    order_str = original_lhs[1:-1]
                    if not order_str.isdigit():
                        raise ValueError("Non-numeric order in D-notation")
                    order = int(order_str)
                
                # Format LHS LaTeX
                if order == 1:
                    lhs_latex = r'\frac{dy}{dt}'
                else:
                    lhs_latex = r'\frac{d^{' + str(order) + r'}y}{dt^{' + str(order) + r'}}'
                
                # Process the RHS (which already has **)
                processed_rhs = rhs 
                try:
                    # Replace Dky terms (using original structure knowledge if needed)
                    temp_rhs_for_d_replace = function_str.split('=', 1)[1].strip() # Use original RHS for D-replacement logic
                    processed_rhs_for_latex = processed_rhs # Keep the version with **
                    
                    for k in range(order, 0, -1):
                        # Perform replacement based on original structure but apply to ** version
                        d_notation = f'D{k}y' if k > 1 else 'D1y'
                        dy_notation = 'Dy'
                        latex_deriv = r'\frac{dy}{dt}' if k == 1 else r'\frac{d^{' + str(k) + r'}y}{dt^{' + str(k) + r'}}'
                        
                        if d_notation in temp_rhs_for_d_replace:
                             processed_rhs_for_latex = processed_rhs_for_latex.replace(d_notation, latex_deriv)
                        if k == 1 and dy_notation in temp_rhs_for_d_replace:
                             processed_rhs_for_latex = processed_rhs_for_latex.replace(dy_notation, latex_deriv)

                    # Replace standalone 'y' with 'y(t)'
                    processed_rhs_for_latex = re.sub(r'\by\b', 'y(t)', processed_rhs_for_latex)
                    
                    # Try SymPy on the processed string (already has **)
                    try:
                        t_sym = sp.symbols('t')
                        y_func = sp.Function('y')(t_sym)
                        expr = sp.sympify(processed_rhs_for_latex, locals={'y': y_func, 't': t_sym})
                        rhs_latex = sp.latex(expr)
                    except (sp.SympifyError, TypeError, SyntaxError):
                        rhs_latex = processed_rhs_for_latex
                    
                    return f"{lhs_latex} = {rhs_latex}"
                except Exception as e:
                    logger.error(f"Error processing RHS for LaTeX (D-notation): {e}", exc_info=True)
                    # Fallback using already processed RHS
                    if order == 1:
                        return r'\frac{dy}{dt} = ' + rhs
                    else:
                        return r'\frac{d^{' + str(order) + r'}y}{dt^{' + str(order) + r'}} = ' + rhs
            except ValueError as ve:
                 logger.warning(f"Could not parse D-notation '{original_lhs}': {ve}")
                 # Fallback to standard SymPy parsing on the ** version
                 try:
                     t_sym = sp.symbols('t')
                     y_func = sp.Function('y')(t_sym)
                     lhs_expr = sp.sympify(lhs, locals={'y': y_func, 't': t_sym})
                     rhs_expr = sp.sympify(rhs, locals={'y': y_func, 't': t_sym})
                     return f"{sp.latex(lhs_expr)} = {sp.latex(rhs_expr)}"
                 except:
                     return safe_function_str # Fallback to ** string
            except Exception as e:
                logger.error(f"Error converting D-notation to LaTeX: {e}", exc_info=True)
                return safe_function_str # Fallback
        
        # For other equation types (non D-notation), use the ** version
        try:
            t_sym = sp.symbols('t')
            y_func = sp.Function('y')(t_sym) 
            lhs_expr = sp.sympify(lhs, locals={'y': y_func, 't': t_sym})
            rhs_expr = sp.sympify(rhs, locals={'y': y_func, 't': t_sym})
            return f"{sp.latex(lhs_expr)} = {sp.latex(rhs_expr)}"
        except:
            return safe_function_str # Fallback to ** string
    
    # For expressions without = sign (use ** version)
    try:
        t_sym = sp.symbols('t')
        y_func = sp.Function('y')(t_sym)
        expr = sp.sympify(safe_function_str, locals={'y': y_func, 't': t_sym})
        return f"y'(t) = {sp.latex(expr)}"
    except:
        return safe_function_str # Fallback to ** string

def create_sidebar():
    """Creates the Streamlit sidebar UI components."""
    with st.sidebar:
        # Simple header
        st.markdown('<h1 class="main-header">MOSES-ODE</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Numerical ODE Solver</p>', unsafe_allow_html=True)
        
        # --- Add Help Button --- 
        def show_help_popup():
            st.session_state.show_help = True
            
        st.button("❓ Help / Input Rules", on_click=show_help_popup, use_container_width=True, help="Show detailed syntax and function rules")
        st.markdown("<hr style='margin-top: 5px; margin-bottom: 10px;'>", unsafe_allow_html=True) # Add a separator
        # --- End Help Button ---
        
        # Tabs for Examples and ODE Library
        st.markdown("**Load Examples or Library ODEs:**") # Add context header
        tab1, tab2 = st.tabs(["Quick Examples", "ODE Library"])
        
        with tab1:
            # Examples section
            selected_example = st.selectbox(
                "Select an example ODE:",
                list(EXAMPLE_ODES.keys()), # Use imported EXAMPLE_ODES
                index=0,
                key="example_selector"
            )
            
            # Use a more descriptive button label
            if st.button("Load Example", key="load_example_btn", use_container_width=True):
                example_data = EXAMPLE_ODES[selected_example]
                # --- Update the correct session state keys linked to input widgets ---
                st.session_state.ode_input = example_data["ode"]
                st.session_state.y0_input = example_data["y0"]
                # t_end uses the value correctly, so keep updating st.session_state.t_end
                st.session_state.t_end = example_data.get("t_end", 10.0) 
                st.session_state.example_loaded = True # Flag to indicate load
                # Trigger a rerun to ensure input widgets refresh immediately
                st.rerun()

        
        with tab2:
            # ODE Library section - categorized
            selected_category = st.selectbox(
                "Select category:",
                list(ODE_LIBRARY.keys()), # Use imported ODE_LIBRARY
                key="library_category"
            )
            
            selected_ode = st.selectbox(
                "Select ODE:",
                list(ODE_LIBRARY[selected_category].keys()), # Use imported ODE_LIBRARY
                key="library_ode"
            )
            
            # Display details concisely
            ode_details = ODE_LIBRARY[selected_category][selected_ode]
            st.caption(f"**Description:** {ode_details.get('description', 'N/A')}")
            st.caption(f"**ODE:** `{ode_details.get('ode', 'N/A')}`")
            st.caption(f"**y0:** `{ode_details.get('y0', 'N/A')}`")
            st.caption(f"**t_end:** {ode_details.get('t_end', 'N/A')}")

            if st.button("Load from Library", key="load_library_btn", use_container_width=True):
                 # --- Update the correct session state keys linked to input widgets ---
                 st.session_state.ode_input = ode_details["ode"]
                 st.session_state.y0_input = ode_details["y0"]
                 # t_end uses the value correctly, so keep updating st.session_state.t_end
                 st.session_state.t_end = ode_details.get("t_end", 10.0) 
                 st.session_state.library_loaded = True # Flag to indicate load
                 # Trigger a rerun to ensure input widgets refresh immediately
                 st.rerun()

        # Add a divider
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Add help/info section (optional)
        with st.expander("About MOSES-ODE"):
            st.markdown("""
            **MOSES-ODE** is a tool for numerically solving Ordinary Differential Equations.
            
            - Define your ODE using standard mathematical notation.
            - Select a solver method.
            - Set initial conditions and time span.
            - Visualize or save the results.
            """)

# Add other UI component functions here as needed, e.g.:
# def create_main_panel_inputs():
#     pass

# def display_results_area(t, y, fig):
#     pass

# --- ODE Solver Tab Components ---

def create_problem_definition_ui():
    """Creates the UI elements for the 'Problem Definition' section."""
    st.markdown("## 1. Problem Definition")
    
    # Initialize session state 
    if "example_loaded" not in st.session_state:
        st.session_state.example_loaded = False
    if "library_loaded" not in st.session_state:
        st.session_state.library_loaded = False
    # Initialize only the key-based state
    if "ode_input" not in st.session_state:
        st.session_state.ode_input = ""
    if "y0_input" not in st.session_state:
        st.session_state.y0_input = ""
    if "show_validation_success" not in st.session_state:
        st.session_state.show_validation_success = False
    # Initialize t_end with default value (5.0)
    if "t_end" not in st.session_state:
        st.session_state.t_end = 5.0
    
    # Enhanced ODE input with more prominent styling
    st.markdown("""
    <style>
    .prominent-input label {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #1E88E5 !important;
    }
    .prominent-input textarea {
        border: 2px solid #1E88E5 !important;
        border-radius: 6px !important;
        font-size: 1.1rem !important;
        font-family: 'Courier New', monospace !important;
        min-height: 60px !important;
        background-color: #f7f9fc !important;
    }
    .latex-preview {
        background-color: #f0f8ff;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 10px;
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

    # ODE Equation Input with validation button
    with st.container():
        st.markdown("### ODE Equation")
        
        # Read directly from the session state key for the preview
        if st.session_state.ode_input:
            latex_formula = convert_to_latex(st.session_state.ode_input)
            st.markdown('<div class="latex-preview">', unsafe_allow_html=True)
            st.latex(latex_formula)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Input field uses key, remove on_change
        st.markdown('<div class="prominent-input">', unsafe_allow_html=True)    
        ode = st.text_area( # Assign to variable ode for consistency, though state is in key
            "Enter your equation",
            key="ode_input", 
            help="For systems use format: [expr1, expr2, ...] Press Ctrl+Enter to apply changes.",
            height=80
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Format help text and validate button
        col1, col2 = st.columns([3, 1])
        with col1:
             st.markdown("""
             <div style="font-size: 0.8rem; color: #555; margin-top: -5px; margin-bottom: 10px">
                 Example format for system: <code>[y1, (1 - y0**2) * y1 - y0]</code>
             </div>
             """, unsafe_allow_html=True)
        
        with col2:
             # Validate equation button
             validate_button = st.button("Validate Equation", use_container_width=True)
            
        if validate_button:
            # Reset success flag at the beginning of validation attempt
            st.session_state.show_validation_success = False 
            try:
                # Use the key state for validation
                ode_input_val = st.session_state.get('ode_input', '') # Use .get for safety
                if not ode_input_val:
                    st.error("Please enter an ODE equation first.")
                    st.stop() # Use st.stop() to halt execution here
                
                # Use the input value from the key for parsing/validation
                ode_for_validation = ode_input_val
                
                # Determine ode_to_parse and num_initial_conditions_needed based on format
                ode_to_parse = ode_for_validation # Default
                is_d_notation = False
                num_initial_conditions_needed = 1 # Default
                
                if '=' not in ode_for_validation:
                    if (ode_for_validation.strip().startswith('[') and ode_for_validation.strip().endswith(']')):
                         # System: [f0, f1, ...]
                         ode_to_parse = ode_for_validation
                         try:
                             num_elements = len(ode_to_parse.strip()[1:-1].split(','))
                             num_initial_conditions_needed = num_elements if num_elements > 0 else 1
                         except Exception: num_initial_conditions_needed = 1 # Fallback
                    else:
                         # Simple expression: y' = f(t, y)
                         ode_to_parse = ode_for_validation
                         num_initial_conditions_needed = 1
                else: # Contains '='
                    parts = ode_for_validation.strip().split('=', 1)
                    lhs = parts[0].strip()
                    rhs = parts[1].strip()

                    if lhs.startswith('D') and 'y' in lhs:
                         is_d_notation = True
                         try:
                             from moses_ode.parsing.symbolic_parser import sys_of_ode
                             converted_system_string = sys_of_ode(ode_for_validation)
                             ode_to_parse = converted_system_string
                             # Determine order for dummy y0 shape
                             if lhs == 'Dy': order = 1
                             elif lhs.startswith('D') and lhs.endswith('y') and lhs[1:-1].isdigit():
                                 order = int(lhs[1:-1])
                             else: order = 1 # Fallback
                             num_initial_conditions_needed = order if order > 0 else 1
                         except Exception as sys_ode_err:
                             st.error(f"Error processing D-notation: {sys_ode_err}")
                             st.stop()
                    elif lhs.lower() in ['y\'', 'dy/dt', 'd1y']:
                         ode_to_parse = rhs
                         num_initial_conditions_needed = 1
                    else:
                         st.error(f"Unrecognized equation format for validation: '{lhs} = ...'. Try solving directly.")
                         st.stop()

                # Try to parse the function string (either converted or original RHS/system)
                with st.spinner("Validating equation syntax..."):
                    # Remove the explicit check for '^' as parsers now handle it via replacement
                    # if '^' in ode_to_parse:
                    #    st.error("Invalid syntax: Use '**' for exponentiation, not '^'.")
                    #    st.stop()

                    f = parse_function_string(ode_to_parse) # This handles ^ -> ** internally via sympify

                    # Test with dummy values 
                    if num_initial_conditions_needed < 1: num_initial_conditions_needed = 1
                    dummy_y0 = np.zeros(num_initial_conditions_needed)
                    
                    try:
                       f_validated = validate_function_input(f, 0.0, dummy_y0)
                       st.session_state.show_validation_success = True # Mark success
                    except Exception as validate_err:
                       logger.error(f"Validation with dummy values failed: {validate_err}") 
                       st.error(f"Validation Error: {validate_err}")
                       st.info("The basic syntax seems okay, but the function might not behave as expected with the required inputs/outputs.")

            except ValueError as ve:
                st.error(f"Validation Error: {str(ve)}")
                st.info("Please check the structure of your D-notation or function.")
            except Exception as e:
                logger.error(f"General validation error: {e}", exc_info=True) 
                st.error(f"Invalid equation syntax or structure: {str(e)}")
                st.info("Check syntax (e.g., use '**' or '^ ' for power) or the overall equation structure.")
        
        # Display persistent success message if flag is set
        if st.session_state.get("show_validation_success", False):
            st.success("Equation syntax appears valid! ✓")
            # Remove the immediate reset to make the message persistent until next validation
            # st.session_state.show_validation_success = False

    # Display loaded example/library info
    if st.session_state.get("example_loaded", False) and "selected_example" in st.session_state:
        st.info("Example loaded. See details in sidebar.")
        st.session_state.example_loaded = False  # Reset flag after display
    elif st.session_state.get("library_loaded", False) and "selected_lib_category" in st.session_state and "selected_lib_ode" in st.session_state:
        st.info("Library ODE loaded. See details in sidebar.")
        st.session_state.library_loaded = False  # Reset flag

    # --- Detect ODE Format for y0 Preview --- 
    ode_input_val = st.session_state.get('ode_input', '').strip()
    ode_format = 'scalar' # Default
    ode_order = 1        # Default
    if '=' not in ode_input_val:
        if ode_input_val.startswith('[') and ode_input_val.endswith(']'):
            ode_format = 'system'
            try: # Estimate dimension
                ode_order = max(1, len(ode_input_val[1:-1].split(',')))
            except: ode_order = 1
        else: 
            ode_format = 'scalar' # Simple expression y'=f(t,y)
            ode_order = 1
    else:
        lhs = ode_input_val.split('=', 1)[0].strip()
        if lhs.startswith('D') and 'y' in lhs:
            ode_format = 'd_notation'
            try: # Determine order
                if lhs == 'Dy': ode_order = 1
                elif lhs[1:-1].isdigit(): ode_order = int(lhs[1:-1])
                else: ode_order = 1 # Fallback if format weird e.g. Dt y
            except: ode_order = 1 
        elif lhs.lower() in ['y\'', 'dy/dt', 'd1y']:
            ode_format = 'scalar' # Explicit first order
            ode_order = 1
        # else: # Unknown format, treat as scalar maybe? Handled by default.
            
    # Display other inputs in a standard 2-column layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Initial Conditions")
        
        # --- Display Context-Aware y0 Preview (Moved Here - Before Input) --- 
        y0_input_val = st.session_state.get("y0_input", "")
        if y0_input_val: # Only show preview if there's input
            current_t0 = st.session_state.get("t0_input", 0.0) # Read t0 state
            y0_values, basic_latex_val = _parse_y0_input(y0_input_val)
            final_latex = ""

            if ode_format == 'd_notation' and len(y0_values) == ode_order:
                cases = []
                for i, val in enumerate(y0_values):
                    deriv_latex = _format_derivative_latex(i, current_t0)
                    cases.append(f"{deriv_latex} = {val}")
                final_latex = r'\begin{cases} ' + r' \\ '.join(cases) + r' \end{cases}'
            elif ode_format == 'system' and len(y0_values) == ode_order: 
                cases = []
                for i, val in enumerate(y0_values):
                    cases.append(f"y_{{{i}}}({current_t0}) = {val}") 
                final_latex = r'\begin{cases} ' + r' \\ '.join(cases) + r' \end{cases}'
            else: 
                final_latex = f"y({current_t0}) = {basic_latex_val}"

            st.markdown('<div class="latex-preview">', unsafe_allow_html=True)
            st.latex(final_latex)
            st.markdown('</div>', unsafe_allow_html=True)
        # --- End Preview ---
            
        # Input field (Now after preview)
        st.markdown('<div class="prominent-input">', unsafe_allow_html=True)
        y0_text = st.text_area(
            "Enter initial values",
            key="y0_input", 
            help="For systems use format: [y0_0, y0_1, ...] Press Ctrl+Enter to apply changes.",
            height=68
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Help text for y0 (Remains after input)
        inner_col1, inner_col2 = st.columns([3, 1])
        with inner_col1:
            st.markdown("""
            <div style="font-size: 0.8rem; color: #555; margin-top: -5px; margin-bottom: 10px">
                Example: <code>0.5</code> (scalar) or <code>[1, 0]</code> (D2y=...) or <code>[3.14, 0]</code> (system)
            </div>
            """, unsafe_allow_html=True)

    with col2:
        # Define t0 input (no change here, state is read above)
        t0 = st.number_input(
            "Initial Time (t0)",
            value=0.0,
            help="Starting time for the simulation",
            step=0.1,
            format="%.2f",
            key="t0_input" 
        )
        # ... (t_end input remains the same) ...
        t_end_min_value = float(st.session_state.get("t0_input", 0.0) + 0.1)
        t_end = st.number_input(
            "End Time (t_end)",
            value=float(st.session_state.get("t_end", 5.0)), 
            min_value=t_end_min_value, 
            help="Ending time for the simulation",
            step=1.0,
            format="%.2f",
            key="t_end_input"
        )

    # Return the key-based state variables 
    return st.session_state.ode_input, st.session_state.y0_input, t0, t_end


def create_solver_config_ui():
    """Creates the UI elements for the 'Solver Configuration' section."""
    st.markdown("## 2. Solver Configuration")
    
    # Add Basic/Advanced mode toggle at the top of the solver config
    if "advanced_mode" not in st.session_state:
        st.session_state.advanced_mode = False
    
    st.markdown("""
    <style>
    .toggle-container {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
    }
    .toggle-label {
        margin-right: 15px;
        font-weight: 500;
    }
    .card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
        background-color: #f9f9f9;
    }
    .card-header {
        font-weight: 500;
        margin-bottom: 10px;
        color: #2e7d32;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Toggle for Basic/Advanced mode
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="toggle-container">', unsafe_allow_html=True)
        st.markdown('<div class="toggle-label">Mode:</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.session_state.advanced_mode = st.toggle(
            "Advanced Mode",
            value=st.session_state.advanced_mode,
            help="Toggle between Basic and Advanced mode to show/hide technical parameters"
        )
    
    # Get the list of solvers from constants
    from moses_ode.app_components.constants import SOLVER_DESCRIPTIONS
    
    # Group solvers by type (requires updates to constants.py later)
    basic_solvers = ["rk4", "euler"]
    adaptive_solvers = ["rk45"]
    implicit_solvers = ["backward_euler", "crank_nicolson"]
    other_solvers = ["heun", "midpoint"]
    
    if st.session_state.advanced_mode:
        # Advanced mode: show all solvers in categorized way
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Solver Method</div>', unsafe_allow_html=True)
        
        method_type = st.radio(
            "Solver Type",
            ["Fixed-step Methods", "Adaptive Methods", "Implicit Methods", "Other Methods"],
            horizontal=True,
            key="solver_type_radio"
        )
        
        if method_type == "Fixed-step Methods":
            solver_list = basic_solvers
        elif method_type == "Adaptive Methods":
            solver_list = adaptive_solvers
        elif method_type == "Implicit Methods":
            solver_list = implicit_solvers
        else:
            solver_list = other_solvers
        
        solver = st.selectbox(
            "Select Method",
            solver_list,
            index=0,
            help="Select the numerical method to use for solving the ODE",
            key="solver_selector"
        )
        
        st.markdown(f'<p class="solver-description">{SOLVER_DESCRIPTIONS[solver]}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)  # Close card
        
        # Parameter settings in expandable section
        with st.expander("Solver Parameters", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                h = st.number_input(
                    "Step Size (h)",
                    value=0.1,
                    min_value=0.001,
                    max_value=1.0,
                    step=0.01,
                    format="%.3f",
                    help="Step size for fixed-step solvers",
                    key="h_input"
                )
            
            with col2:
                tol = st.number_input(
                    "Tolerance (tol)",
                    value=1e-6,
                    min_value=1e-12,
                    max_value=1e-3,
                    format="%.2e",
                    help="Tolerance for adaptive-step solvers (only used by adaptive methods like RK45)",
                    key="tol_input"
                )
                
            # Add advanced options (can add more here later)
            st.markdown("##### Additional Options")
            max_steps = st.slider(
                "Maximum Steps",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                help="Maximum number of steps before terminating the solution process"
            )
    else:
        # Basic mode: simplified interface with limited options
        col1, col2 = st.columns(2)
        
        with col1:
            solver = st.selectbox(
                "Numerical Method",
                ["rk4", "euler", "rk45"],  # Only show basic methods
                index=0,
                format_func=lambda x: {
                    "rk4": "Runge-Kutta 4 (Recommended)",
                    "euler": "Euler (Simple)",
                    "rk45": "Adaptive RK45 (Variable step)"
                }.get(x, x),
                help="Select the numerical method to use for solving the ODE",
                key="solver_selector"
            )
            st.markdown(f'<p class="solver-description">{SOLVER_DESCRIPTIONS[solver]}</p>', unsafe_allow_html=True)
        
        with col2:
            h = st.number_input(
                "Step Size",
                value=0.1,
                min_value=0.001,
                max_value=1.0,
                step=0.01,
                format="%.3f",
                help="Step size for the solver (smaller values give more accurate results but take longer)",
                key="h_input"
            )
            
            # Only show tolerance for adaptive methods
            if solver == "rk45":
                tol = st.number_input(
                    "Tolerance",
                    value=1e-6,
                    format="%.2e",
                    help="Accuracy tolerance for adaptive solvers",
                    key="tol_input"
                )
            else:
                tol = 1e-6  # Default value for fixed-step methods
    
    # Return the selected solver and parameters
    return solver, h, tol


def create_output_options_ui():
    """Creates the UI elements for the 'Output Options' section."""
    st.markdown("## 3. Output Options")
    
    # Use st.expander for collapsibility
    with st.expander("Configure Output and Plotting", expanded=True): # Start expanded
        # Remove the card layout divs
        # st.markdown('<div class="card">', unsafe_allow_html=True)
        # st.markdown('<div class="card-header">Result Options</div>', unsafe_allow_html=True)
        
        # Display options in two columns inside the expander
        col1, col2 = st.columns(2)

        with col1:
            plot_results = st.checkbox("Plot Results", value=True, key="plot_results_checkbox")
            
            # Only show plot options if plot_results is checked
            if plot_results:
                plot_type = st.radio(
                    "Plot Type",
                    ["Line", "Scatter", "Phase Portrait (for systems)"],
                    index=0,
                    key="plot_type_radio"
                )
            else:
                plot_type = "Line"  # Default value

        with col2:
            save_results = st.checkbox("Save Results", value=False, key="save_results_checkbox")
            
            # Show file format options when save_results is checked
            if save_results:
                file_format = st.selectbox(
                    "File Format",
                    ["CSV", "Excel", "JSON"],
                    index=0,
                    help="Select the format for saving the results"
                )
            else:
                file_format = "CSV"  # Default value
                
            max_points = st.slider(
                "Maximum Points", 
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                help="Limit the number of points to display and save for better performance",
                key="max_points_slider"
            )
        
        # Remove the closing card div
        # st.markdown('</div>', unsafe_allow_html=True)  # Close card
    
    # Set default values for plot options that were previously in advanced display options
    # Keep this outside the expander if it controls behavior triggered elsewhere
    st.session_state.advanced_plot_options = {
        "color": "#1E88E5",
        "grid": True,
        "legend_pos": "Auto",
        "dpi": 100,
        "size": "Medium",
        "width": 10.0,
        "height": 6.0,
        "file_format": file_format if 'file_format' in locals() else "CSV" # Ensure file_format is accessible
    }
    
    return plot_results, plot_type, save_results, max_points 

# Add helper for derivative LaTeX
def _format_derivative_latex(order, var_value):
    """Generates LaTeX string for y^(order)(var_value)."""
    if order == 0:
        return f"y({var_value})"
    elif order == 1:
        return f"y'({var_value})"
    else:
        return f"y^{{({order})}}({var_value})" # Using y^{(n)}(t) notation

# Refactor initial conditions parser/formatter
def _parse_y0_input(y0_str):
    """
    Parses the initial conditions string into a list of value strings.
    Also returns a basic LaTeX formatted version of the value(s).
    
    Args:
        y0_str: Initial conditions string
        
    Returns:
        Tuple: (list_of_value_strings, basic_latex_value_format)
               Returns (None, y0_str) if parsing fails significantly.
    """
    if not y0_str or not y0_str.strip():
        return [], "" # Empty list, empty string
    
    cleaned = y0_str.strip()
    values = []
    basic_latex = cleaned # Default fallback

    # Check if y0 is an array-like format (heuristic)
    is_list_format = (',') in cleaned or (cleaned.startswith('[') and cleaned.endswith(']'))

    if is_list_format:
        # Remove brackets if present for splitting
        if cleaned.startswith('[') and cleaned.endswith(']'):
            content = cleaned[1:-1].strip()
        else:
            content = cleaned
        
        if content: # Avoid splitting an empty string
            values = [v.strip() for v in content.split(',')]
            # Try to format as pmatrix if successful
            try:
                basic_latex = r'\begin{pmatrix} ' + r' \\ '.join(values) + r' \end{pmatrix}'
            except:
                 basic_latex = f"[{content}]" # Fallback LaTeX for list
        else: # Input was '[]'
             values = []
             basic_latex = r'\begin{pmatrix} \end{pmatrix}'

    else: # Single value
        values = [cleaned] # List with one element
        basic_latex = cleaned # LaTeX is just the value itself
        
    # Return the list of individual value strings and the basic formatted value
    return values, basic_latex


# Rename the old function to avoid breaking calls if missed (can remove later)
def format_initial_conditions_latex(y0_str):
     _, basic_latex = _parse_y0_input(y0_str)
     return basic_latex 