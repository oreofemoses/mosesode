import streamlit as st
import time
from moses_ode.app_components.ui import create_problem_definition_ui, create_solver_config_ui, create_output_options_ui
from moses_ode.app_components.solver_logic import solve_ode, prepare_results_dataframe, create_solution_plots

def create_wizard_ui():
    """Create a wizard-style step-by-step interface for the ODE solver."""
    
    # Initialize session state for wizard
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 1
    if "wizard_data" not in st.session_state:
        st.session_state.wizard_data = {}
    
    # Add CSS for wizard steps
    st.markdown("""
    <style>
    .wizard-container {
        margin-bottom: 20px;
    }
    .wizard-steps {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
        position: relative;
    }
    .wizard-steps::before {
        content: '';
        position: absolute;
        top: 15px;
        left: 0;
        right: 0;
        height: 2px;
        background: #e0e0e0;
        z-index: 1;
    }
    .step {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background-color: #e0e0e0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        position: relative;
        z-index: 2;
    }
    .step.active {
        background-color: #1E88E5;
        color: white;
    }
    .step.completed {
        background-color: #4CAF50;
        color: white;
    }
    .step-label {
        text-align: center;
        margin-top: 8px;
        font-size: 0.8rem;
        color: #616161;
    }
    .navigation-buttons {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
    }
    .wizard-content {
        background-color: #f9f9f9;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Wizard progress bar
    st.markdown('<div class="wizard-container">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    # Render step indicators
    step_html = '<div class="wizard-steps">'
    steps = ["Define Problem", "Configure Solver", "Set Output Options", "Results"]
    
    for i, step_name in enumerate(steps, 1):
        if i < st.session_state.wizard_step:
            step_class = "completed"
        elif i == st.session_state.wizard_step:
            step_class = "active"
        else:
            step_class = ""
            
        step_html += f'''
        <div>
            <div class="step {step_class}">{i}</div>
            <div class="step-label">{step_name}</div>
        </div>
        '''
    
    step_html += '</div>'
    st.markdown(step_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Wizard content
    st.markdown('<div class="wizard-content">', unsafe_allow_html=True)
    
    # Step 1: Problem Definition
    if st.session_state.wizard_step == 1:
        st.markdown("### Step 1: Define Your Ordinary Differential Equation")
        ode, y0, t0, t_end = create_problem_definition_ui()
        
        # Store values in session state
        st.session_state.wizard_data["ode"] = ode
        st.session_state.wizard_data["y0"] = y0
        st.session_state.wizard_data["t0"] = t0
        st.session_state.wizard_data["t_end"] = t_end
        
        # Navigation
        st.markdown('<div class="navigation-buttons">', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Next: Configure Solver", use_container_width=True):
                if not ode or not y0:
                    st.error("Please provide both ODE equation and initial conditions")
                else:
                    st.session_state.wizard_step = 2
    
    # Step 2: Solver Configuration
    elif st.session_state.wizard_step == 2:
        st.markdown("### Step 2: Configure the Solver")
        solver, h, tol = create_solver_config_ui()
        
        # Store values in session state
        st.session_state.wizard_data["solver"] = solver
        st.session_state.wizard_data["h"] = h
        st.session_state.wizard_data["tol"] = tol
        
        # Navigation
        st.markdown('<div class="navigation-buttons">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.wizard_step = 1
        with col2:
            if st.button("Next: Set Output Options", use_container_width=True):
                st.session_state.wizard_step = 3
    
    # Step 3: Output Options
    elif st.session_state.wizard_step == 3:
        st.markdown("### Step 3: Configure Output Options")
        plot_results, plot_type, save_results, max_points = create_output_options_ui()
        
        # Store values in session state
        st.session_state.wizard_data["plot_results"] = plot_results
        st.session_state.wizard_data["plot_type"] = plot_type
        st.session_state.wizard_data["save_results"] = save_results
        st.session_state.wizard_data["max_points"] = max_points
        
        # Navigation
        st.markdown('<div class="navigation-buttons">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.wizard_step = 2
        with col2:
            if st.button("Solve ODE", use_container_width=True):
                # All data collected, proceed to solving
                st.session_state.wizard_step = 4
    
    # Step 4: Results
    elif st.session_state.wizard_step == 4:
        st.markdown("### Step 4: Results")
        
        # Extract saved data
        ode = st.session_state.wizard_data.get("ode", "")
        y0 = st.session_state.wizard_data.get("y0", "")
        t0 = st.session_state.wizard_data.get("t0", 0.0)
        t_end = st.session_state.wizard_data.get("t_end", 10.0)
        solver = st.session_state.wizard_data.get("solver", "rk45")
        h = st.session_state.wizard_data.get("h", 0.1)
        tol = st.session_state.wizard_data.get("tol", 1e-6)
        plot_results = st.session_state.wizard_data.get("plot_results", True)
        plot_type = st.session_state.wizard_data.get("plot_type", "Line")
        save_results = st.session_state.wizard_data.get("save_results", False)
        max_points = st.session_state.wizard_data.get("max_points", 1000)
        
        # Show a summary of inputs
        with st.expander("ODE Problem Summary", expanded=False):
            st.code(f"ODE: {ode}")
            st.code(f"Initial condition: {y0}")
            st.code(f"Time span: t ∈ [{t0}, {t_end}]")
            st.code(f"Solver: {solver}")
            st.code(f"Step size: {h}, Tolerance: {tol}")
        
        # Check if solution is already calculated in this session
        if "solution_computed" not in st.session_state.wizard_data:
            # Progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_status(message, progress_percent):
                status_text.text(message)
                progress_bar.progress(progress_percent / 100)
                time.sleep(0.1)  # Small delay for visual feedback
            
            # Solve the ODE
            try:
                update_status("Starting ODE solver...", 10)
                t, y, solve_time, metadata = solve_ode(
                    ode=ode, 
                    y0=y0, 
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
                
                # Store solution data
                st.session_state.wizard_data["solution_computed"] = True
                st.session_state.wizard_data["solution_t"] = t
                st.session_state.wizard_data["solution_y"] = y
                st.session_state.wizard_data["solution_df"] = df
                st.session_state.wizard_data["solution_metadata"] = metadata
                st.session_state.wizard_data["solution_y_dim"] = y_dim
                st.session_state.wizard_data["has_nan"] = has_nan
                st.session_state.wizard_data["has_inf"] = has_inf
                st.session_state.wizard_data["rows_removed"] = rows_removed
                
                # Complete progress
                update_status("Done!", 100)
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()
                
                # Force Streamlit to rerun to display the results
                st.experimental_rerun()
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Check your inputs and try again.")
        
        else:
            # Display results from already computed solution
            df = st.session_state.wizard_data.get("solution_df")
            y_dim = st.session_state.wizard_data.get("solution_y_dim")
            metadata = st.session_state.wizard_data.get("solution_metadata")
            solve_time = metadata.get("compute_time", 0)
            has_nan = st.session_state.wizard_data.get("has_nan", False)
            has_inf = st.session_state.wizard_data.get("has_inf", False)
            rows_removed = st.session_state.wizard_data.get("rows_removed", 0)
            
            # Show warnings if needed
            if has_nan:
                st.warning("Warning: NaN values detected in the raw solution.")
            if has_inf:
                st.warning("Warning: Inf values detected in the raw solution.")
            if rows_removed > 0:
                st.warning(f"Removed {rows_removed} rows containing NaN/Inf values for plotting/display.")
            
            if df.empty:
                st.error("No valid data points remaining after filtering NaN/Inf values. Cannot proceed.")
            else:
                # Performance metrics
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                with metrics_col1:
                    st.metric("Computation Time", f"{solve_time:.4f} s")
                with metrics_col2:
                    st.metric("Steps Taken", metadata.get("steps", 0))
                with metrics_col3:
                    st.metric("Average Step Size", f"{metadata.get('avg_step_size', 0):.5f}")
                with metrics_col4:
                    st.metric("Dimension", f"{y_dim}")
                
                # Results table
                st.subheader("Solution Data")
                st.dataframe(df, use_container_width=True)
                
                # Plotting results
                if plot_results:
                    st.subheader("Solution Plot")
                    
                    # Create plot
                    fig, plot_successful, plot_info = create_solution_plots(df, y_dim, plot_type, solver)
                    
                    # Display plot info
                    st.text(f"{plot_info}")
                    
                    # Display the plot
                    if plot_successful:
                        try:
                            st.pyplot(fig)
                            import matplotlib.pyplot as plt
                            plt.close(fig)
                        except Exception as plot_err:
                            st.error(f"Error displaying plot: {str(plot_err)}")
                            import matplotlib.pyplot as plt
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
        
        # Navigation
        st.markdown('<div class="navigation-buttons">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Options", use_container_width=True):
                st.session_state.wizard_step = 3
        with col2:
            if st.button("Start Over", use_container_width=True):
                # Reset the wizard
                st.session_state.wizard_step = 1
                st.session_state.wizard_data = {}
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close navigation buttons
    st.markdown('</div>', unsafe_allow_html=True)  # Close wizard content 