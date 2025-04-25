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

def main():
    # Create the sidebar using the imported function
    create_sidebar()
    
    # Main content
    st.markdown('<h1 style="text-align: center;">Ordinary Differential Equation Solver</h1>', unsafe_allow_html=True)
    
    # Initialize interface mode in session state
    # if "wizard_mode" not in st.session_state: # Removed Wizard Mode toggle initialization
    #     st.session_state.wizard_mode = False
    
    # Add interface mode toggle
    # st.markdown('<div class="mode-toggle-container">', unsafe_allow_html=True) # Removed Wizard Mode toggle UI
    # col1, col2, col3 = st.columns([2, 1, 2])
    # with col2:
    #     st.session_state.wizard_mode = st.toggle(
    #         "Wizard Mode",
    #         value=st.session_state.wizard_mode,
    #         help="Toggle between classic interface and step-by-step wizard"
    #     )
    # st.markdown('</div>', unsafe_allow_html=True)
    
    # Create main tabs
    ode_solver_tab, numerical_tools_tab = st.tabs(["ODE Solver", "Numerical Tools"])
    
    # Numerical Methods tab - using the imported function
    with numerical_tools_tab:
        create_numerical_methods_tab()
    
    # ODE Solver tab
    with ode_solver_tab:
        # if st.session_state.wizard_mode: # Removed Wizard Mode conditional logic
        #     # Use the wizard interface
        #     create_wizard_ui()
        # else:
        # Use the classic interface (always now)
        # Problem Definition using the new UI function
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
                # Validate required fields
                if not ode or not y0:
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
                
                # Call the solver logic function
                update_status("Starting ODE solver...", 10)
                t, y, solve_time, metadata = solve_ode(
                    ode=ode, 
                    y0_str=y0,
                    t0=t0, 
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
                st.error(f"An error occurred: {str(e)}")
                st.info("Check your inputs and try again. Make sure the ODE and initial conditions are properly formatted.")
        
        # Footer
        st.markdown('<div class="footer">MOSES-ODE: A Numerical Ordinary Differential Equation Solver</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 