import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import json # Import json for safer parsing
from moses_ode.parsing.function_parser import parse_function_string
from moses_ode.parsing.symbolic_parser import sys_of_ode
from moses_ode.validation.input_validation import validate_function_input
from moses_ode.solvers import euler, rk4, rk45, backward_euler, crank_nicolson, heun, midpoint
import streamlit as st

def solve_ode(ode, y0_str, t0, t_end, solver="rk45", h=0.1, tol=1e-6, status_callback=None):
    """
    Solves an ODE with the specified method and parameters.
    
    Args:
        ode (str): The ODE string to solve
        y0_str (str): Initial conditions string (e.g., "[0, 1]")
        t0 (float): Start time
        t_end (float): End time
        solver (str): Solver method to use
        h (float): Step size for fixed-step solvers
        tol (float): Tolerance for adaptive solvers
        status_callback (function): Optional callback for status updates
    
    Returns:
        tuple: (t, y, solve_time, metadata) where t and y are arrays of time and solution points,
               solve_time is computation time, and metadata is a dict with additional info
    """
    # Update status if callback provided
    if status_callback:
        status_callback("Parsing ODE...", 20)
    
    # Determine ODE structure and parse function f
    is_system = False
    expected_y0_dim = 1 # Default for simple ODEs
    try:
        # Check if input is already a system in list format
        ode_stripped = ode.strip()
        if ode_stripped.startswith('[') and ode_stripped.endswith(']'):
            if status_callback:
                status_callback("Parsing system of ODEs (list format)...", 30)
            # Count elements in the provided list string
            try:
                # Use json.loads to parse the list structure first
                parsed_list = json.loads(ode_stripped)
                if isinstance(parsed_list, list):
                     expected_y0_dim = len(parsed_list)
                else: # Handle case where input is like '[scalar]'
                     expected_y0_dim = 1 
            except json.JSONDecodeError as json_err:
                # Fallback: simple comma splitting if JSON fails (less robust)
                print(f"Warning: JSON parsing failed for list input ({json_err}). Falling back to comma split.")
                expected_y0_dim = len([item for item in ode_stripped.strip('[]').split(',') if item.strip()])

            f = parse_function_string(ode_stripped) # Parse the original list string
            is_system = True
        elif 'D' in ode: # Check for D-notation if not a list
            if status_callback:
                status_callback("Parsing system of ODEs (D-notation)...", 30)
            sys_output_str = sys_of_ode(ode)
            # Determine expected dimension from the CONVERTED system output string
            try:
                 # Use json.loads to parse the list structure first
                 parsed_list = json.loads(sys_output_str)
                 if isinstance(parsed_list, list):
                      expected_y0_dim = len(parsed_list)
                 else: # Handle case where input is like '[scalar]'
                      expected_y0_dim = 1
            except json.JSONDecodeError as json_err:
                 # Fallback: simple comma splitting if JSON fails (less robust)
                 print(f"Warning: JSON parsing failed for D-notation output ({json_err}). Falling back to comma split.")
                 expected_y0_dim = len([item for item in sys_output_str.strip('[]').split(',') if item.strip()])
            
            f = parse_function_string(sys_output_str) # Parse the converted string
            is_system = True
        else: # Otherwise, assume it's a simple scalar ODE f(t,y) for y'=f(t,y)
            if status_callback:
                status_callback("Parsing single ODE...", 30)
            expected_y0_dim = 1 
            f = parse_function_string(ode) # Parse the expression directly
    except Exception as e:
        raise ValueError(f"Error parsing ODE function: {str(e)}") from e
        
    # Parse and validate initial conditions y0
    try:
        # Use json.loads for safer parsing than eval
        y0_list = json.loads(y0_str)
        y0_parsed = np.array(y0_list, dtype=float)
        # Ensure it's at least 1D
        if y0_parsed.ndim == 0: 
            y0_parsed = y0_parsed.reshape((1,))
    except Exception as e:
        raise ValueError(f"Error parsing Initial Conditions 'y(t0)': Input should be a number or a list/tuple like [0, 1]. Error: {str(e)}") from e

    # Validate the dimension of y0 against the expected dimension from the ODE
    if y0_parsed.size != expected_y0_dim:
        ode_type_str = f"{expected_y0_dim}-element system (derived from {ode})" if is_system else "scalar ODE"
        raise ValueError(
            f"Initial condition dimension mismatch: ODE ({ode_type_str}) requires {expected_y0_dim} initial value(s), but received {y0_parsed.size} ({y0_str})."
        )
        
    # Validate the parsed function f against the parsed initial conditions
    try:
        f = validate_function_input(f, t0, y0_parsed)
    except Exception as e:
        # Catch validation errors specifically if possible, otherwise re-raise general error
        raise ValueError(f"ODE function validation failed: {str(e)}") from e
    
    if status_callback:
        status_callback(f"Solving with {solver} method...", 40)
    
    # Solve ODE with selected method
    start_time = time.time()
    
    if solver == "euler":
        t, y = euler.euler_solver(f, t0, y0_parsed, h, t_end)
    elif solver == "rk4":
        t, y = rk4.rk4_solver(f, t0, y0_parsed, h, t_end)
    elif solver == "rk45":
        t, y = rk45.rk45_solver(f, t0, y0_parsed, h, t_end, tol)
        # Handle special case for RK45 when it returns a collection of 0-d arrays
        if len(y.shape) == 2 and y0_parsed.size == 1 and y.shape[1] != 1:
            # Reshape if needed to handle the case where each row is a scalar
            y = y.reshape(y.shape[0], 1) if y.shape[1] != 1 else y
    elif solver == "backward_euler":
        t, y = backward_euler.backward_euler_solver(f, t0, y0_parsed, h, t_end)
    elif solver == "crank_nicolson":
        t, y = crank_nicolson.crank_nicolson_solver(f, t0, y0_parsed, h, t_end)
    elif solver == "heun":
        t, y = heun.heun_solver(f, t0, y0_parsed, h, t_end)
    elif solver == "midpoint":
        t, y = midpoint.midpoint_solver(f, t0, y0_parsed, h, t_end)
    else:
        raise ValueError(f"Unknown solver method: {solver}")
    
    # Calculate solve time
    solve_time = time.time() - start_time
    
    # Check and process the solver output
    if y.ndim == 1:  # If y is a 1D array (simple scalar solution)
        pass  # Keep as is
    elif y.ndim == 2 and y.shape[1] == 1:  # 2D array but only one column
        y = y.flatten()  # Flatten to 1D array
    elif y.ndim == 2 and y0_parsed.size == 1:
        # For scalar ODE that somehow returned a 2D array
        # We'll try to extract the correct column
        y = y.reshape(-1, 1)[:, 0]  # Take the first column
    
    # Determine dimension of the solution
    if y.ndim == 1:
        y_dim = 1
    else:
        y_dim = y.shape[1]  # Number of variables in system
    
    # Return results
    metadata = {
        "solver": solver,
        "h": h,
        "tol": tol,
        "t0": t0,
        "t_end": t_end,
        "y_dim": y_dim,
        "compute_time": solve_time,
        "steps": len(t),
        "avg_step_size": (t_end - t0) / len(t)
    }
    
    return t, y, solve_time, metadata


def prepare_results_dataframe(t, y, max_points=1000):
    """
    Converts the ODE solution arrays into a clean pandas DataFrame.
    
    Args:
        t (array): Time points
        y (array): Solution points (1D for scalar ODEs, 2D for systems)
        max_points (int): Maximum number of points to include (for downsampling)
    
    Returns:
        tuple: (df, y_dim) where df is the pandas DataFrame and y_dim is the dimension of the solution
    """
    # Format results as DataFrame
    if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
        # Scalar ODE case (or system that returned flattened output)
        y_flat = y.flatten()
        df = pd.DataFrame({"t": t, "y0": y_flat})  # Label as y0 for consistency
        y_dim = 1
    elif y.ndim == 2:  # System of ODEs
        result_dict = {"t": t}
        y_dim = y.shape[1]  # Number of state variables (e.g., 2 for D2y)
        for i in range(y_dim):
            result_dict[f"y{i}"] = y[:, i]  # Use y0, y1, y2... labels
        df = pd.DataFrame(result_dict)
    else:
        raise ValueError(f"Unexpected solution shape: {y.shape}")
    
    # Check for NaN/Inf values
    has_nan = np.isnan(y).any()
    has_inf = np.isinf(y).any()
    
    # Downsample if needed
    if len(df) > max_points:
        df = df.iloc[np.linspace(0, len(df)-1, max_points, dtype=int)]
        df = df.reset_index(drop=True)  # Reset index after downsampling
    
    # Clean up data - remove NaN/Inf values
    initial_rows = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    final_rows = len(df)
    rows_removed = initial_rows - final_rows
    
    return df, y_dim, has_nan, has_inf, rows_removed


def create_solution_plots(df, y_dim, plot_type, solver):
    """
    Creates plots of the ODE solution.
    
    Args:
        df (DataFrame): DataFrame with time and solution values
        y_dim (int): Dimension of the solution (1 for scalar, >1 for systems)
        plot_type (str): Type of plot to create ("Line", "Scatter", or "Phase Portrait")
        solver (str): Name of the solver used (for plot title)
    
    Returns:
        tuple: (fig, plot_successful, plot_info) with the matplotlib figure, success flag, and info message
    """
    # Use simplified default values for plot properties
    plot_color = "#1E88E5"  # Default blue color
    show_grid = True
    legend_loc = 'best'
    plot_dpi = 100
    figsize = (10, 6)  # Medium size
    
    # Create figure with specified size
    fig, ax = plt.subplots(figsize=figsize, dpi=plot_dpi)
    plot_successful = False
    plot_info = ""
    
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    try:
        if plot_type == "Line":
            if y_dim == 1:
                y_column = "y0" if "y0" in df.columns else df.columns[1]
                if not df.empty:
                    # Extract data and ensure finiteness
                    t_plot = df["t"].values
                    y_plot = df[y_column].values
                    assert np.all(np.isfinite(t_plot)), "Non-finite t data detected before plotting!"
                    assert np.all(np.isfinite(y_plot)), f"Non-finite {y_column} data detected before plotting!"
                    
                    ax.plot(t_plot, y_plot, "-", linewidth=2, label="y(t)", color=plot_color)
                    ax.autoscale(enable=True, axis='y', tight=False)
                    plot_info = "Plotting y(t) with line plot"
                    plot_successful = True
            else:  # System
                # Use default colormap for multiple lines
                import matplotlib.cm as cm
                colors = cm.viridis(np.linspace(0, 1, y_dim))
                
                for i in range(y_dim):
                    y_column = f"y{i}"
                    if y_column in df.columns and not df.empty:
                        # Extract data and ensure finiteness
                        t_plot = df["t"].values
                        y_plot = df[y_column].values
                        assert np.all(np.isfinite(t_plot)), f"Non-finite t data detected before system line {i}!"
                        assert np.all(np.isfinite(y_plot)), f"Non-finite {y_column} data detected before system line {i}!"
                        
                        ax.plot(t_plot, y_plot, "-", linewidth=2, label=f"y{i}(t)", color=colors[i])
                        plot_successful = True
                
                # Only show legend for system plots
                ax.legend(loc=legend_loc)
                plot_info = f"Plotting {y_dim}-dimensional system with line plot"
        
        elif plot_type == "Scatter":
            if y_dim == 1:
                y_column = "y0" if "y0" in df.columns else df.columns[1]
                if not df.empty:
                    t_plot = df["t"].values
                    y_plot = df[y_column].values
                    ax.scatter(t_plot, y_plot, s=15, alpha=0.7, label="y(t)", color=plot_color)
                    ax.autoscale(enable=True, axis='y', tight=False)
                    plot_info = "Plotting y(t) with scatter plot"
                    plot_successful = True
            else:  # System
                # Use default colormap for multiple lines
                import matplotlib.cm as cm
                colors = cm.viridis(np.linspace(0, 1, y_dim))
                
                for i in range(y_dim):
                    y_column = f"y{i}"
                    if y_column in df.columns and not df.empty:
                        t_plot = df["t"].values
                        y_plot = df[y_column].values
                        ax.scatter(t_plot, y_plot, s=15, alpha=0.7, label=f"y{i}(t)", color=colors[i])
                        plot_successful = True
                
                ax.legend(loc=legend_loc)
                plot_info = f"Plotting {y_dim}-dimensional system with scatter plot"
        
        elif plot_type == "Phase Portrait (for systems)":
            if y_dim >= 2:  # Need at least 2 dimensions for phase portrait
                if "y0" in df.columns and "y1" in df.columns and not df.empty:
                    y0_plot = df["y0"].values
                    y1_plot = df["y1"].values
                    assert np.all(np.isfinite(y0_plot)), "Non-finite y0 data detected before phase portrait!"
                    assert np.all(np.isfinite(y1_plot)), "Non-finite y1 data detected before phase portrait!"
                    
                    # Color the points based on time (gradient)
                    times = df["t"].values
                    points = ax.scatter(y0_plot, y1_plot, c=times, s=15, cmap='viridis', 
                                       alpha=0.7, label="Phase Portrait")
                    plt.colorbar(points, ax=ax, label="Time")
                    
                    # Add arrows to show direction
                    n_points = len(y0_plot)
                    arrow_indices = np.linspace(0, n_points-2, min(20, n_points//5), dtype=int)
                    for i in arrow_indices:
                        ax.annotate("", xy=(y0_plot[i+1], y1_plot[i+1]), xytext=(y0_plot[i], y1_plot[i]),
                                   arrowprops=dict(arrowstyle="->", color=plot_color, lw=1.5))
                    
                    ax.set_xlabel("y0")
                    ax.set_ylabel("y1")
                    plot_info = "Plotting phase portrait (y1 vs y0)"
                    plot_successful = True
                else:
                    plot_info = "Cannot create phase portrait: missing y0 or y1 data"
            else:
                plot_info = "Cannot create phase portrait: need at least 2 dimensions"
        
        # Common settings for all plots
        if plot_successful:
            # Set grid based on preference
            ax.grid(show_grid)
            
            # Add title and labels
            ax.set_title(f"ODE Solution using {solver} method", fontsize=12)
            ax.set_xlabel("Time (t)", fontsize=10)
            ax.set_ylabel("Solution y(t)", fontsize=10)
            
            # Add background color and style enhancements
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#f8f9fa')
            
            # Add light box around plot area
            for spine in ax.spines.values():
                spine.set_edgecolor('#cccccc')
        
    except Exception as e:
        plot_info = f"Error creating plot: {str(e)}"
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))  # Create empty plot
        ax.text(0.5, 0.5, f"Plot Error: {str(e)}", ha='center', va='center')
        return fig, False, plot_info
    
    return fig, plot_successful, plot_info 