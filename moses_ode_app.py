import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from moses_ode.parsing.function_parser import parse_function_string
from moses_ode.parsing.symbolic_parser import sys_of_ode
from moses_ode.validation.input_validation import validate_function_input
# Import constants from the new module
from moses_ode.app_components.constants import SOLVER_DESCRIPTIONS, EXAMPLE_ODES, ODE_LIBRARY
# Import UI components
from moses_ode.app_components.ui import create_sidebar, create_problem_definition_ui, create_solver_config_ui, create_output_options_ui
# Import Numerical Methods tab UI
from moses_ode.app_components.numerical_methods_ui import create_numerical_methods_tab
# Import ODE solver logic
from moses_ode.app_components.solver_logic import solve_ode, prepare_results_dataframe, create_solution_plots
# Import Wizard UI
# from moses_ode.app_components.wizard_ui import create_wizard_ui # Removed Wizard UI

# Set page configuration
st.set_page_config(
    page_title="MOSES-ODE Solver",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    /* Simplified styles */
    .main-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1E88E5;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 0.9rem;
        color: #455A64;
        margin-top: 0;
        margin-bottom: 15px;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 500;
        color: #37474F;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .info-card {
        background-color: #e8f5e9;
        border-left: 4px solid #43a047;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .solver-description {
        font-size: 0.85rem;
        color: #546e7a;
        font-style: italic;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .divider {
        border-top: 1px solid #e0e0e0;
        margin: 1.5rem 0;
    }
    .results-section {
        margin-top: 2rem;
    }
    .footer {
        text-align: center;
        font-size: 0.8rem;
        color: #78909c;
        margin-top: 3rem;
    }
    /* Mode toggle container */
    .mode-toggle-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
    }
    .mode-toggle-label {
        margin-right: 15px;
        font-weight: 500;
        color: #455A64;
    }
    /* Hide Streamlit branding and hamburger menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Help Display Function --- 
def display_help_section():
    """Displays the detailed help content in the main area."""
    st.markdown("## Input Rules and Syntax Help")
    
    def hide_help_popup():
        st.session_state.show_help = False
        
    st.button("❌ Close Help", on_click=hide_help_popup)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    ### General Rules
    *   **Exponentiation:** Use `**` (e.g., `x**2`, `y**3`). or **use** `^`.
    *   **Variables:**
        *   **ODE Solver:** Use `t` for time and `y` (or `y0`, `y1`, etc. for systems/higher-order) for the dependent variable(s).
        *   **Numerical Tools:** Use `x` for the independent variable.
    *   **Constants:** Use `pi` for π and `E` for Euler's number `e`.
    *   **Functions:** Standard mathematical functions are supported:
        *   Trigonometric: `sin()`, `cos()`, `tan()`, `asin()`, `acos()`, `atan()`, `atan2()`
        *   Hyperbolic: `sinh()`, `cosh()`, `tanh()`
        *   Exponential/Logarithmic: `exp()`, `log()` (natural log), `log10()`, `log2()`
        *   Roots/Powers: `sqrt()`, `cbrt()`
        *   Other: `abs()`
    *   **NumPy (`np.`) Prefix:** Generally **not** required or supported in the input boxes. Use the standard function names (e.g., `sin(t)` instead of `np.sin(t)`).

    ---        
    ### ODE Solver Tab
    
    **1. ODE Equation Input:**
    
    *   **Scalar First-Order (y' = f(t, y)):** Enter the expression for `f(t, y)`. 
        *   Example: `-0.5 * y + exp(t/2)`
    *   **System of First-Order (y' = f(t, y)):** Enter a list of expressions separated by commas, enclosed in square brackets `[]`. The variables must be `t`, `y0`, `y1`, ..., `y(n-1)` where `n` is the number of equations.
        *   Example (Pendulum): `[y1, -9.81/0.5 * sin(y0)]` (Here, `y0` is angle θ, `y1` is angular velocity ω)
    *   **Higher-Order (Dⁿy = f(t, y, Dy, ..., Dⁿ⁻¹y)):** Enter the equation using `D` notation for derivatives.
        *   `Dy` or `D1y` for the first derivative (y')
        *   `D2y` for the second derivative (y'')
        *   `Dny` for the n-th derivative (y⁽ⁿ⁾)
        *   Example (Damped Oscillator): `D2y = -0.1*D1y - 5*y`
        *   **Note:** The solver internally converts this to a system using `y0=y`, `y1=Dy`, `y2=D2y`, etc.

    **2. Initial Conditions (y0):**
    
    *   **Scalar:** Enter a single numerical value.
        *   Example: `1.5`
    *   **System / Higher-Order:** Enter a list of numerical values in square brackets `[]`, corresponding to `y0(t0)`, `y1(t0)`, ..., `y(n-1)(t0)`.
        *   The number of values **must match** the dimension of the system or the order of the ODE.
        *   Example (Pendulum System): `[3.14, 0]` (Initial angle π, initial velocity 0)
        *   Example (Damped Oscillator D2y=...): `[1, 0]` (Initial position 1, initial velocity 0)

    ---        
    ### Numerical Tools Tab
    
    **1. Function Input (f(x)):**
    
    *   Enter a function of a single variable `x`.
    *   Example (Integration): `x**2 * exp(-x)`
    *   Example (Differentiation): `sin(x) / x`
    
    """)

def main():
    # --- Initialize Session State --- 
    if 'show_help' not in st.session_state:
        st.session_state.show_help = False
    # Initialize other states if needed (e.g., for inputs if not using keys properly)
    # ... (ensure ode_input, y0_input etc. are initialized if create_*_ui relies on them existing)
    if "ode_input" not in st.session_state: st.session_state.ode_input = ""
    if "y0_input" not in st.session_state: st.session_state.y0_input = ""
    # ... add for numerical tools keys if necessary ...
    if "integration_function" not in st.session_state: st.session_state.integration_function = "x^2"
    if "differentiation_function" not in st.session_state: st.session_state.differentiation_function = "sin(x)"
        
    # Create the sidebar using the imported function
    create_sidebar()
    
    # Main content area
    st.markdown('<h1 style="text-align: center;">Ordinary Differential Equation Solver</h1>', unsafe_allow_html=True)
    
    # --- Conditional Display: Help or Main App --- 
    if st.session_state.get('show_help', False):
        display_help_section()
    else:
        # --- Display Main App --- 
        # Create main tabs
        ode_solver_tab, numerical_tools_tab = st.tabs(["ODE Solver", "Numerical Tools"])
        
        # Numerical Methods tab - using the imported function
        with numerical_tools_tab:
            create_numerical_methods_tab()
        
        # ODE Solver tab
        with ode_solver_tab:
            # Problem Definition using the new UI function
            # Ensure returned values are handled (even if state is primary)
            ode, y0, t0, t_end = create_problem_definition_ui()
    
            # Solver Configuration using the new UI function
            solver, h, tol = create_solver_config_ui()
    
            # Output Options using the new UI function
            plot_results, plot_type, save_results, max_points = create_output_options_ui()
    
            # Solve Button
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
            solve_col1, solve_col2, solve_col3 = st.columns([1, 2, 1])
            with solve_col2:
                solve_button = st.button("Solve ODE", use_container_width=True)
    
            # Results Section
            if solve_button:
                try:
                    # Validate required fields (read from state for robustness)
                    ode_val = st.session_state.get('ode_input', '')
                    y0_val = st.session_state.get('y0_input', '')
                    if not ode_val or not y0_val:
                        st.error("ODE and initial conditions are required!")
                        st.stop()
                    
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Define status callback for progress updates
                    def update_status(message, progress_percent):
                        status_text.text(message)
                        progress_bar.progress(progress_percent / 100)
                        time.sleep(0.1)  # Small delay for visual feedback
                    
                    # Call the solver logic function using state values
                    update_status("Starting ODE solver...", 10)
                    t, y, solve_time, metadata = solve_ode(
                        ode=ode_val, 
                        y0_str=y0_val,
                        t0=t0, # t0, t_end likely don't need state if keys aren't used
                        t_end=t_end, 
                        solver=solver, 
                        h=h, 
                        tol=tol,
                        status_callback=update_status
                    )
                    
                    # Get solution dimension
                    y_dim = metadata["y_dim"]
                    
                    update_status("Processing results...", 80)
                    
                    # Prepare DataFrame with results
                    df, y_dim, has_nan, has_inf, rows_removed = prepare_results_dataframe(t, y, max_points)
                    
                    # Show warnings if needed
                    if has_nan:
                        st.warning("Warning: NaN values detected in the raw solution.")
                    if has_inf:
                        st.warning("Warning: Inf values detected in the raw solution.")
                    if rows_removed > 0:
                        st.warning(f"Removed {rows_removed} rows containing NaN/Inf values for plotting/display.")
                    
                    if df.empty:
                        st.error("No valid data points remaining after filtering NaN/Inf values. Cannot proceed.")
                        st.stop()
                    
                    # Complete progress
                    update_status("Done!", 100)
                    time.sleep(0.5)
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Display results
                    st.markdown('<div class="results-section">', unsafe_allow_html=True)
                    st.markdown("## 4. Results")
                    
                    # Performance metrics
                    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                    with metrics_col1:
                        st.metric("Computation Time", f"{solve_time:.4f} s")
                    with metrics_col2:
                        st.metric("Steps Taken", metadata["steps"])
                    with metrics_col3:
                        st.metric("Average Step Size", f"{metadata['avg_step_size']:.5f}")
                    with metrics_col4:
                        st.metric("Dimension", f"{y_dim}")
                    
                    # Show table
                    st.markdown("### Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Plotting results
                    if plot_results:
                        st.markdown("### Solution Plot")
                        
                        # Create plot
                        fig, plot_successful, plot_info = create_solution_plots(df, y_dim, plot_type, solver)
                        
                        # Display plot info
                        st.text(f"{plot_info}")
                        
                        # Display the plot
                        if plot_successful:
                            try:
                                st.pyplot(fig)
                                plt.close(fig)
                            except Exception as plot_err:
                                st.error(f"Error displaying plot: {str(plot_err)}")
                                plt.close(fig)
                        else:
                            st.info("Plotting skipped: No valid data available or plotting failed.")
                    
                    # Save results option
                    if save_results:
                        # Get the file format from advanced options
                        file_format = st.session_state.advanced_plot_options.get("file_format", "CSV")
                        
                        if file_format == "CSV":
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name=f"ode_solution_{solver}.csv",
                                mime="text/csv"
                            )
                        elif file_format == "Excel":
                            import io
                            output = io.BytesIO()
                            with st.spinner("Preparing Excel file..."):
                                df.to_excel(output, index=False, engine='openpyxl')
                                output.seek(0)
                            st.download_button(
                                label="Download Excel",
                                data=output,
                                file_name=f"ode_solution_{solver}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        elif file_format == "JSON":
                            json_str = df.to_json(orient="records")
                            st.download_button(
                                label="Download JSON",
                                data=json_str,
                                file_name=f"ode_solution_{solver}.json",
                                mime="application/json"
                            )
                    
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"An error occurred during solving or processing: {str(e)}")
                    # Log the full error for debugging
                    import traceback
                    st.code(traceback.format_exc())
                    # Clean up progress indicators on error
                    if 'status_text' in locals() and status_text:
                        status_text.empty()
                    if 'progress_bar' in locals() and progress_bar:
                        progress_bar.empty()
        
    # Footer (consider moving inside the 'else' block if you don't want it on help page)
    st.markdown("---        ")
    st.markdown('<div class="footer">MOSES-ODE v0.1.0 | Developed with Streamlit</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 