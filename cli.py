import click
import numpy as np
import json # Added for safer parsing
from typing import List, Callable, Tuple, Any
from moses_ode.solvers import euler, rk4, rk45, backward_euler, crank_nicolson, heun, midpoint  # Import solvers
from moses_ode.parsing.function_parser import parse_function_string
from moses_ode.parsing.symbolic_parser import sys_of_ode
from moses_ode.validation.input_validation import validate_function_input
from moses_ode.benchmark_problems import get_problem, PROBLEMS # Added PROBLEMS import
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt # Moved import to top

# Initialize rich console for styled output
console = Console()

# --- Helper Functions ---

def parse_y0_string(y0_str: str) -> np.ndarray:
    """Safely parse the initial condition string (scalar or list) into a NumPy array."""
    try:
        # Attempt to parse as JSON (handles lists like [1, 0] and scalars like 1.0)
        parsed_y0 = json.loads(y0_str)
        # Ensure it's a numpy array of floats
        return np.array(parsed_y0, dtype=float).flatten() # Flatten ensures shape (n,) or ()
    except json.JSONDecodeError:
        # Handle simple scalar case if not valid JSON (e.g., just '1')
        try:
            return np.array([float(y0_str)], dtype=float)
        except ValueError:
            raise ValueError(f"Could not parse initial condition: '{y0_str}'. Use format '1.0' or '[1.0, 0.0]'.")
    except Exception as e:
         raise ValueError(f"Error parsing initial condition '{y0_str}': {e}")


_SOLVER_MAP = {
    "euler": euler.euler_solver,
    "rk4": rk4.rk4_solver,
    "rk45": rk45.rk45_solver,
    "backward_euler": backward_euler.backward_euler_solver,
    "crank_nicolson": crank_nicolson.crank_nicolson_solver,
    "heun": heun.heun_solver,
    "midpoint": midpoint.midpoint_solver,
}

def dispatch_solver(
    solver_name: str,
    f: Callable,
    t0: float,
    y0: np.ndarray,
    t_end: float,
    h: float | None = None,
    tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """Calls the appropriate solver based on its name."""
    solver_func = _SOLVER_MAP.get(solver_name)
    if not solver_func:
        raise ValueError(f"Unknown solver: {solver_name}")

    fixed_step_solvers = ["euler", "rk4", "backward_euler", "crank_nicolson", "heun", "midpoint"]
    adaptive_step_solvers = ["rk45"] # Add others if available

    if solver_name in fixed_step_solvers:
        if h is None:
            raise ValueError(f"Fixed-step solver '{solver_name}' requires step size 'h'.")
        return solver_func(f, t0, y0, h, t_end)
    elif solver_name in adaptive_step_solvers:
        # Use provided h as initial step guess if available, otherwise default (e.g., 0.1)
        initial_h = h if h is not None else 0.1
        # Ensure tolerance is passed
        if solver_name == "rk45": # Specific check for rk45 signature
             return solver_func(f, t0, y0, initial_h, t_end, tol)
        else:
             # Adapt if other adaptive solvers have different signatures
             # return solver_func(f, t0, y0, initial_h, t_end, tol=tol) # Example
             raise NotImplementedError(f"Adaptive solver '{solver_name}' dispatch not fully implemented.")
    else:
        # This case should ideally not be reached due to the initial check
        raise ValueError(f"Solver '{solver_name}' is neither fixed nor adaptive in dispatch map.")


# --- CLI Commands ---

@click.group()
def cli():
    """MOSES-ODE: Numerical ODE Solver (CLI)"""
    pass


@cli.command()
@click.option("--ode", required=True, help="ODE string (e.g., \"y' = -2*y + sin(t)\" or for systems \"[y[1], -y[0]]\")") # Updated help
@click.option("--solver",
              type=click.Choice(list(_SOLVER_MAP.keys())), # Use keys from map
              default="rk45", help="Solver method")
@click.option("--t0", type=float, default=0.0, help="Initial time")
@click.option("--y0", required=True, help="Initial condition(s) (e.g., '1.0' or '[0.0, 1.0]')") # Updated help
@click.option("--t_end", type=float, required=True, help="End time")
@click.option("--h", type=float, help="Step size (fixed-step solvers, initial guess for adaptive)") # Updated help
@click.option("--tol", type=float, default=1e-6, help="Tolerance (adaptive solvers)")
@click.option("--plot", is_flag=True, help="Plot results")
@click.option("--output", type=click.Path(), help="Save results to CSV")
def solve(ode, solver, t0, y0, t_end, h, tol, plot, output):
    """Solve an ODE with a specified numerical method."""
    try:
        # Parse initial conditions safely
        y0_parsed = parse_y0_string(y0)

        # Parse ODE string into a callable function
        # Handle potential 'D' notation conversion first
        if 'D' in ode:
            # Assuming sys_of_ode handles 'D' notation and returns a compatible string
            ode_str_parsed = sys_of_ode(ode)
            console.print(f"[bold blue]sys_of_ode output:[/bold blue] {ode_str_parsed}")
        else:
            ode_str_parsed = ode

        # Now parse the final string (original or from sys_of_ode)
        f = parse_function_string(ode_str_parsed)

        # Validate function signature against y0
        f = validate_function_input(f, t0, y0_parsed) # Keep validation

        # Dispatch to solver using the helper function
        t, y = dispatch_solver(solver, f, t0, y0_parsed, t_end, h, tol)

        # Display results in a table
        console.print(f"\n[bold green]Solution using {solver}:[/bold green]")
        table = Table(title="Solution Summary (First 10 points)")
        table.add_column("Time (t)", justify="right", style="cyan")
        # Add columns dynamically based on y shape
        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            # Handle true scalar or vector that happens to be 1D result
            y_disp = y.flatten() # Ensure it's 1D for display
            table.add_column("y(t)", justify="right", style="magenta")
            for ti, yi in zip(t[:10], y_disp[:10]):
                table.add_row(f"{ti:.4f}", f"{yi:.4f}")
        elif y.ndim == 2:
            # Handle system with multiple components
            for i in range(y.shape[1]):
                 table.add_column(f"y_{i}(t)", justify="right", style="magenta")
            for i in range(min(10, len(t))): # Iterate up to 10 rows
                row_data = [f"{t[i]:.4f}"] + [f"{y[i, j]:.4f}" for j in range(y.shape[1])]
                table.add_row(*row_data)
        else:
             console.print("[yellow]Warning: Cannot display solution with >2 dimensions in table.[/yellow]")

        console.print(table)

        # Save or plot results
        if output:
            # Save numerical data robustly
            header = "t," + ",".join([f"y_{i}" for i in range(y.shape[1])]) if y.ndim==2 else "t,y"
            data_to_save = np.column_stack((t, y)) if y.ndim==2 else np.column_stack((t, y))
            np.savetxt(output, data_to_save, delimiter=",", header=header, comments="")
            console.print(f"Results saved to [bold green]{output}[/bold green]")
        if plot:
            # Use matplotlib for plotting
            plt.figure(figsize=(10, 6)) # Slightly larger plot
            plt.plot(t, y, label=f'y(t)' if y.ndim==1 else [f'y_{i}(t)' for i in range(y.shape[1])])
            if y.ndim == 2 and y.shape[1] > 1:
                 plt.legend() # Add legend for systems
            plt.xlabel("Time (t)")
            plt.ylabel("Solution (y)")
            plt.title(f"Solution using {solver} for ODE: {ode}")
            plt.grid(True)
            plt.show()

        return 0  # Success

    except (ValueError, TypeError) as e: # Catch specific expected errors
         console.print(f"[bold red]Input Error: {e}[/bold red]")
         return 1 # Return error code
    except ImportError:
         console.print("[bold red]Error: Matplotlib is required for plotting. Install with 'pip install matplotlib'.[/bold red]")
         return 1
    except Exception as e:
        console.print(f"[bold red]Runtime Error: {e}[/bold red]")
        # Consider logging the full traceback here for debugging
        # import traceback
        # # console.print(traceback.format_exc())
        return 1  # Return error code


@cli.command()
@click.option("--problem", type=click.Choice(list(PROBLEMS.keys())), default="stiff",
              help="Type of predefined benchmark ODE problem") # Use keys from imported problems
@click.option("--solvers", required=True, help="Comma-separated list of solvers (e.g., 'euler,rk45')")
@click.option("--t_end", type=float, default=None, help="End time (overrides problem default)")
@click.option("--h", type=float, default=0.01, help="Default step size for fixed-step solvers") # Added default h
@click.option("--tol", type=float, default=1e-6, help="Tolerance (adaptive solvers)")
def benchmark(problem, solvers, t_end, h, tol):
    """Benchmark MOSES-ODE solvers on a predefined problem."""
    try:
        # Get problem definition
        problem_def = get_problem(problem)
        ode_str = problem_def['ode_str']
        y0_str = problem_def['y0_str']
        t0, default_t_end = problem_def['t_span']
        analytical_sol = problem_def['analytical_sol']

        # Override t_end if provided
        t_end = t_end if t_end is not None else default_t_end

        # Parse initial conditions safely
        y0_parsed = parse_y0_string(y0_str)

        # Parse ODE string
        f = parse_function_string(ode_str)
        f = validate_function_input(f, t0, y0_parsed) # Validate

        # Parse solvers list
        solver_list = [s.strip() for s in solvers.split(",") if s.strip()]
        valid_solvers = list(_SOLVER_MAP.keys())

        # Check for invalid solvers
        invalid_solvers = [s for s in solver_list if s not in valid_solvers]
        if invalid_solvers:
            console.print(f"[bold red]Error: Invalid solver(s): {', '.join(invalid_solvers)}[/bold red]")
            console.print(f"Available solvers: {', '.join(valid_solvers)}")
            return 1  # Return error code

        # Benchmark each solver
        console.print(f"\n[bold green]Benchmarking on '{problem}' problem (t=[{t0}, {t_end}]):[/bold green]")
        table = Table(title=f"Benchmark Results ({problem} problem)")
        table.add_column("Solver", justify="left", style="cyan")
        table.add_column("Steps", justify="right", style="magenta")
        table.add_column("Max Error (vs Analytical)", justify="right", style="yellow")
        table.add_column("Status", justify="left", style="green") # Added status column

        for solver_name in solver_list:
            status = "OK"
            error_val = np.nan
            num_steps = "N/A"
            try:
                # Dispatch to solver using the helper function
                # Use provided h for fixed-step, tol for adaptive
                t, y = dispatch_solver(solver_name, f, t0, y0_parsed, t_end, h, tol)
                num_steps = str(len(t))

                # Compute error if analytical solution is available
                if analytical_sol:
                    y_exact = analytical_sol(t)
                    # Handle scalar vs vector analytical solution comparison
                    if y.ndim == 1 and y_exact.ndim == 1:
                        error_val = np.max(np.abs(y - y_exact))
                    elif y.ndim == 2 and y_exact.ndim == 1: # Compare against first component if analytical is scalar
                        error_val = np.max(np.abs(y[:, 0] - y_exact))
                        console.print(f"[grey]Note: Comparing {solver_name} output y[:,0] against analytical solution.[/grey]")
                    elif y.ndim == 2 and y_exact.ndim == 2 and y.shape == y_exact.shape:
                         error_val = np.max(np.abs(y - y_exact))
                    else:
                        console.print(f"[yellow]Warning: Cannot compute error for {solver_name} due to shape mismatch between numerical ({y.shape}) and analytical ({getattr(y_exact, 'shape', 'scalar')}).[/yellow]")
                        error_val = np.nan

            except Exception as e:
                status = f"Error: {e}"
                error_val = np.nan
                num_steps = "N/A"
                console.print(f"[bold red]Error benchmarking {solver_name}: {e}[/bold red]")

            # Add row to table
            error_str = f"{error_val:.3e}" if not np.isnan(error_val) else "N/A"
            status_style = "red" if "Error" in status else "green"
            table.add_row(solver_name, num_steps, error_str, f"[{status_style}]{status}[/{status_style}]")


        console.print(table)
        return 0  # Success

    except (ValueError, TypeError) as e: # Catch specific expected errors
         console.print(f"[bold red]Input Error: {e}[/bold red]")
         return 1 # Return error code
    except Exception as e:
        console.print(f"[bold red]Runtime Error: {e}[/bold red]")
        # import traceback
        # # console.print(traceback.format_exc())
        return 1  # Return error code


if __name__ == "__main__":
    cli()