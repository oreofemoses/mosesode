import streamlit as st
# Import constants needed for the sidebar (e.g., example lists)
from moses_ode.app_components.constants import EXAMPLE_ODES, ODE_LIBRARY
from moses_ode.parsing.function_parser import parse_function_string
from moses_ode.validation.input_validation import validate_function_input
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

def create_sidebar():
    """Creates the Streamlit sidebar UI components."""
    with st.sidebar:
        # Simple header
        st.markdown('<h1 class="main-header">MOSES-ODE</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Numerical ODE Solver</p>', unsafe_allow_html=True)
        
        # Tabs for Examples and ODE Library
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
                st.session_state.ode = example_data["ode"]
                st.session_state.y0 = example_data["y0"]
                # Optionally load t_end if defined in EXAMPLE_ODES, provide default
                st.session_state.t_end = example_data.get("t_end", 10.0) 
                st.session_state.example_loaded = True # Flag to indicate load

        
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
                 st.session_state.ode = ode_details["ode"]
                 st.session_state.y0 = ode_details["y0"]
                 st.session_state.t_end = ode_details.get("t_end", 10.0) # Default t_end if not specified
                 st.session_state.example_loaded = True # Flag to indicate load

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
    
    # Initialize session state for the example loading (ensure these exist)
    if "example_loaded" not in st.session_state:
        st.session_state.example_loaded = False
    if "library_loaded" not in st.session_state:
        st.session_state.library_loaded = False
    if "ode" not in st.session_state:
        st.session_state.ode = ""
    if "y0" not in st.session_state:
        st.session_state.y0 = ""
    if "show_validation_success" not in st.session_state:
        st.session_state.show_validation_success = False

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
    </style>
    """, unsafe_allow_html=True)

    # ODE Equation Input with validation button
    with st.container():
        st.markdown('<div class="prominent-input">', unsafe_allow_html=True)
        ode = st.text_area(
            "ODE Equation",
            value=st.session_state.ode,
            help="For systems use format: [expr1, expr2, ...] where expr1, expr2 are functions of y0, y1, etc.",
            key="ode_input",
            height=80
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
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
                    if not ode:
                        st.error("Please enter an ODE equation first.")
                        st.stop() # Use st.stop() to halt execution here

                    ode_to_parse = ode # Start with the original string
                    is_d_notation = False
                    num_initial_conditions_needed = 1 # Default for simple y'=f(t,y)

                    # Basic check for = sign
                    if '=' not in ode:
                         # If it's not a system string '[...]', assume it's just f(t,y) for y' = f(t,y)
                         if not (ode.strip().startswith('[') and ode.strip().endswith(']')):
                              ode_to_parse = ode # Parse the expression directly
                              num_initial_conditions_needed = 1
                         # If it IS a system string, parse it directly
                         else:
                             # Estimate dimension from the system string for dummy y0
                             try:
                                 # Simple split by comma, might be fragile for complex expressions inside
                                 num_elements = len(ode.strip()[1:-1].split(','))
                                 num_initial_conditions_needed = num_elements
                             except Exception as parse_err:
                                 logger.warning(f"Could not reliably determine system dimension for validation: {parse_err}")
                                 num_initial_conditions_needed = 1 # Fallback
                             ode_to_parse = ode

                    else: # Contains '='
                        parts = ode.strip().split('=', 1)
                        lhs = parts[0].strip()
                        rhs = parts[1].strip()

                        if lhs.startswith('D') and 'y' in lhs:
                             is_d_notation = True
                             # Use the fixed sys_of_ode to convert
                             from moses_ode.parsing.symbolic_parser import sys_of_ode # Import locally or at top
                             converted_system_string = sys_of_ode(ode)
                             ode_to_parse = converted_system_string
                             # Determine order for dummy y0 shape
                             try:
                                 order_str = lhs.split('D')[1].split('y')[0]
                                 num_initial_conditions_needed = int(order_str)
                             except Exception as order_err:
                                  logger.warning(f"Could not parse order from D-notation for validation: {order_err}")
                                  num_initial_conditions_needed = 1 # Fallback
                        # Allow common first derivative notations
                        elif lhs.lower() in ['y\'', 'dy/dt', 'd1y']: 
                             ode_to_parse = rhs # Parse only the RHS
                             num_initial_conditions_needed = 1
                        else:
                             # If LHS is not D{N}y or y' etc., and it's not a system [], assume invalid format for validation scope
                             st.error(f"Unrecognized equation format for simple validation: '{lhs} = ...'. Try solving directly.")
                             st.stop()


                    # Try to parse the function string (either converted or original RHS/system)
                    with st.spinner("Validating equation syntax..."):
                        time.sleep(0.5) # Small delay for UX

                        # Check for power operator '^' before parsing
                        if '^' in ode_to_parse:
                            st.error("Invalid syntax: Use '**' for exponentiation, not '^'.")
                            st.stop()

                        f = parse_function_string(ode_to_parse)

                        # Test with dummy values of the estimated correct shape
                        # Ensure dummy_y0 has at least one element
                        if num_initial_conditions_needed < 1: 
                            logger.warning("Estimated needed initial conditions is less than 1, defaulting to 1.")
                            num_initial_conditions_needed = 1 
                        dummy_y0 = np.zeros(num_initial_conditions_needed) 
                        
                        # Wrap validate_function_input in try-except as it might fail even if parse succeeds
                        try:
                           f_validated = validate_function_input(f, 0.0, dummy_y0)
                           st.session_state.show_validation_success = True # Mark success
                        except Exception as validate_err:
                           # Log the specific validation error for debugging
                           logger.error(f"Validation with dummy values failed: {validate_err}") 
                           st.error(f"Validation Error: {validate_err}")
                           st.info("The basic syntax seems okay, but the function might not behave as expected with the required inputs/outputs.")


                except ValueError as ve: # Catch specific ValueErrors (e.g., from sys_of_ode)
                    st.error(f"Validation Error: {str(ve)}")
                    st.info("Please check the structure of your D-notation ODE.")
                except Exception as e:
                    # General parsing error
                    logger.error(f"General validation error: {e}", exc_info=True) # Log full traceback
                    st.error(f"Invalid equation syntax or structure: {str(e)}")
                    st.info("Check your syntax (e.g., use '**' for power) or the overall equation structure.")
        
        # Use get() for safer access to session state
        if st.session_state.get("show_validation_success", False):
            st.success("Equation syntax appears valid! ✓")
            # Reset after showing - important!
            st.session_state.show_validation_success = False

    # Display loaded example/library info
    if st.session_state.get("example_loaded", False) and "selected_example" in st.session_state:
        st.info("Example loaded. See details in sidebar.")
        st.session_state.example_loaded = False  # Reset flag after display
    elif st.session_state.get("library_loaded", False) and "selected_lib_category" in st.session_state and "selected_lib_ode" in st.session_state:
        st.info("Library ODE loaded. See details in sidebar.")
        st.session_state.library_loaded = False  # Reset flag

    # Display other inputs in a standard 2-column layout
    col1, col2 = st.columns(2)

    with col1:
        y0 = st.text_input(
            "Initial Condition (y0)",
            value=st.session_state.y0,
            help="Enter initial condition(s):\n- For scalar ODEs: '1' \n- For systems: '[0, 5]' \n- For Dny (nth-order): provide n values representing [y(0), y'(0), ..., y^(n-1)(0)]",
            key="y0_input"
        )

        st.markdown("""
        <div style="font-size: 0.8rem; color: #555; margin-top: -5px; margin-bottom: 10px">
            For D2y (second-order), provide two values: <code>[y(0), y'(0)]</code>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        t0 = st.number_input(
            "Initial Time (t0)",
            value=0.0,
            help="Starting time for the simulation",
            step=0.1,
            format="%.2f",
            key="t0_input"
        )

        # Initialize t_end session state if not present
        if "t_end" not in st.session_state:
            st.session_state.t_end = 10.0

        t_end = st.number_input(
            "End Time (t_end)",
            value=float(st.session_state.t_end),  # Ensure it's float
            min_value=float(t0 + 0.1),  # Ensure it's float
            help="Ending time for the simulation",
            step=1.0,
            format="%.2f",
            key="t_end_input"
        )
    # Return the values entered by the user
    return ode, y0, t0, t_end


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