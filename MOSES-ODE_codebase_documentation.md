# MOSES-ODE: Mathematical ODE Solver Ecosystem

## Overview

MOSES-ODE is a comprehensive numerical Ordinary Differential Equation (ODE) solver library implemented in Python. The project provides a variety of numerical methods for solving initial value problems (IVPs) for ODEs, ranging from simple first-order methods like Euler's method to more sophisticated adaptive step-size methods like RK45. In addition, it includes numerical methods for differentiation, integration, and various mathematical applications.

The name MOSES-ODE stands for "Mathematical ODE Solver Ecosystem", reflecting its goal of providing a complete ecosystem for ODE solving with consistent interfaces, validation, and visualization capabilities.

## Project Structure

The project is organized into the following main components:

```
moses_ode/
├── parsing/               # Parsing of ODE functions and expressions
│   ├── function_parser.py # Converts string representations to callable functions
│   └── symbolic_parser.py # Converts higher-order ODEs to first-order systems
├── solvers/               # Implementation of various ODE solvers
│   ├── solver_base.py     # Abstract base class for all solvers
│   ├── euler.py           # Euler's method
│   ├── rk4.py             # 4th-order Runge-Kutta method
│   ├── rk45.py            # Adaptive Runge-Kutta method
│   ├── heun.py            # Heun's method (predictor-corrector)
│   ├── midpoint.py        # Midpoint method
│   ├── backward_euler.py  # Implicit Backward Euler method
│   ├── backward_euler_adaptive.py  # Adaptive Backward Euler
│   ├── crank_nicolson.py  # Crank-Nicolson method
│   └── adams_moulton4.py  # 4th-order Adams-Moulton method
├── validation/            # Input validation and error checking
│   └── input_validation.py # Validates ODE functions and inputs
└── numerical/             # Numerical methods beyond ODE solving
    ├── integration.py     # Numerical integration methods
    ├── differentiation.py # Numerical differentiation methods
    └── applications.py    # Practical applications of numerical methods
└── api/                   # API interfaces

# Other files
moses_ode_app.py           # Streamlit web application interface
cli.py                     # Command-line interface (CLI)
```

## Architectural Design

MOSES-ODE follows a modular architecture with clear separation of concerns:

1. **Parsing Layer:** Converts user input (string representations of ODEs) into a format the solver can process
2. **Validation Layer:** Ensures input correctness and compatibility
3. **Solver Layer:** Contains the numerical algorithms for solving ODEs
4. **Numerical Layer:** Provides additional numerical methods for various mathematical operations
5. **Interface Layer:** Provides different ways to interact with the solvers (CLI, web app)

This layered approach allows the core solving algorithms to remain independent of the input method, while making the system easily extensible.

## Component Details

### 1. Parsing Module

The parsing module is responsible for converting string representations of ODEs into callable functions that can be used by the solvers.

#### `function_parser.py`

This module handles the parsing of mathematical expressions into Python functions:

- **Main Function:** `parse_function_string(func_str)`
  - Takes a string representation of an ODE function
  - Returns a callable function of the form f(t, y)
  - Handles both scalar ODEs and systems of ODEs
  - Uses SymPy for secure parsing (prevents code injection)
  - Converts expressions to NumPy functions for efficient numerical computation

Design choices:
- Using SymPy for parsing allows for secure evaluation of user-provided mathematical expressions
- The resulting function is always vectorized to handle both scalar and array inputs
- Error handling is comprehensive to provide clear feedback on parsing errors
- The parser can handle both explicit f(t,y) formats and system formats using array notation

#### `symbolic_parser.py`

This module handles the conversion of higher-order ODEs to systems of first-order ODEs:

- **Main Functions:**
  - `convert_to_first_order(ode, y, t)`: Converts a symbolic nth-order ODE into a system of first-order ODEs
  - `sys_of_ode(ode_str)`: Parses a string representation of a higher-order ODE and returns a string representation of the equivalent first-order system

Design choices:
- The symbolic representation allows for mathematical manipulation of the ODEs
- The higher-order to first-order conversion follows the standard substitution approach where each derivative becomes a new variable
- Special handling for the notation "Dny" indicating the nth derivative of y

### 2. Validation Module

The validation module ensures that the input functions and parameters meet the requirements of the solvers.

#### `input_validation.py`

- **Main Functions:**
  - `validate_function_input(f, t_val, y_val)`: Validates that a callable ODE function returns the expected shape and type
  - `validate_jacobian_input(jac, t_val, y_val)`: Validates Jacobian functions for implicit methods

Design choices:
- Early validation prevents cryptic errors during solver execution
- The validation is performed once before the main solver loop, improving performance
- Returns a wrapped function that ensures consistent output regardless of input format

### 3. Solvers Module

The solvers module contains the implementation of various numerical methods for solving ODEs.

#### `solver_base.py`

This module defines an abstract base class for all ODE solvers:

- **Class:** `ODESolver`
  - Defines a common interface for all solver implementations
  - Handles common operations like solution storage, event detection, error checking
  - Provides utility methods for dense output interpolation

Design choices:
- Object-oriented design with inheritance promotes code reuse
- Abstract base class ensures all solvers follow a consistent interface
- Common functionality is implemented once in the base class

#### Explicit Methods

##### `euler.py`

- Implements the simplest first-order method
- Function: `euler_solver(f, t0, y0, h, t_end, events=None)`

##### `rk4.py`

- Implements the classical 4th-order Runge-Kutta method
- Function: `rk4_solver(f, t0, y0, h, t_end, events=None)`

##### `rk45.py`

- Implements an adaptive step-size Runge-Kutta-Fehlberg method
- Function: `rk45_solver(f, t0, y0, h, t_end, tol=1e-6, events=None)`

##### `heun.py`

- Implements Heun's method (a second-order predictor-corrector)
- Function: `heun_solver(f, t0, y0, h, t_end, events=None)`

##### `midpoint.py`

- Implements the midpoint method (second-order)
- Function: `midpoint_solver(f, t0, y0, h, t_end, events=None)`

#### Implicit Methods

##### `backward_euler.py`

- Implements the first-order implicit Backward Euler method
- Function: `backward_euler_solver(f, t0, y0, h, t_end, jac=None, newton_tol=1e-8, max_iter=100, events=None)`

##### `backward_euler_adaptive.py`

- Implements an adaptive step-size version of the Backward Euler method
- Function: `backward_euler_adaptive_solver(f, t0, y0, h, t_end, tol=1e-6, jac=None, newton_tol=1e-8, max_iter=100, events=None)`

##### `crank_nicolson.py`

- Implements the second-order implicit Crank-Nicolson method
- Function: `crank_nicolson_solver(f, t0, y0, h, t_end, jac=None, newton_tol=1e-8, max_iter=100, events=None)`

##### `adams_moulton4.py`

- Implements the 4th-order implicit Adams-Moulton method
- Function: `adams_moulton4_solver(f, t0, y0, h, t_end, jac=None, newton_tol=1e-8, max_iter=100, events=None)`

Design choices for solvers:
- All solvers follow a consistent interface for interchangeability
- Each solver is implemented in its own module for clarity and maintainability
- Fixed-step methods take a step size h, while adaptive methods automatically adjust h based on error estimates
- Implicit methods include options for custom Jacobian functions and Newton iteration parameters
- Error control in adaptive methods is based on both absolute and relative tolerances

### 4. Numerical Module

The numerical module provides additional mathematical methods beyond ODE solving.

#### `integration.py`

This module implements numerical integration methods:

- **Functions:**
  - `trapezoid(f, a, b, n=100)`: Trapezoidal rule for numerical integration
  - `simpson_1_3(f, a, b, n=100)`: Simpson's 1/3 rule for numerical integration
  - `simpson_3_8(f, a, b, n=99)`: Simpson's 3/8 rule for numerical integration
  - `romberg(f, a, b, max_iterations=10, tol=1e-10)`: Romberg integration with error tolerance

Design choices:
- Comprehensive error handling and validation
- Adaptive convergence in Romberg integration
- Consistent interface across all integration methods

#### `differentiation.py`

This module implements numerical differentiation methods:

- **Functions:**
  - `forward_difference(f, x, h=1e-5)`: Forward difference approximation of first derivative
  - `backward_difference(f, x, h=1e-5)`: Backward difference approximation of first derivative
  - `central_difference(f, x, h=1e-5)`: Central difference approximation of first derivative
  - `second_derivative(f, x, h=1e-5)`: Approximation of second derivative

Design choices:
- Multiple methods with different accuracy characteristics
- Automatic step size adjustment for improved accuracy
- Error handling for edge cases

#### `applications.py`

This module provides practical applications of numerical methods:

- **Functions:**
  - `estimate_ode_coefficients(data_points, order=1)`: Estimates coefficients of ODEs from data
  - `integrate_function_to_solve_ode(f, a, b, y0, n=1000, method='simpson')`: Solves simple ODEs through direct integration
  - `estimate_ode_from_solution(t_points, y_values, order=1)`: Estimates ODE model from solution data

Design choices:
- Integration of various numerical techniques
- Practical applications for data analysis
- Support for both first and second-order systems

## User Interfaces

The project provides two main interfaces for interacting with the solvers:

### Command Line Interface (`cli.py`)

The CLI uses the Click library to provide a user-friendly command-line interface:

- **Commands:**
  - `solve`: Solves a single ODE with specified parameters
  - `benchmark`: Compares multiple solvers on predefined test problems

Design choices:
- Commands follow a consistent pattern with reasonable defaults
- Rich library is used for formatted terminal output
- Support for saving results to CSV and plotting with matplotlib

### Web Application (`moses_ode_app.py`)

A Streamlit-based web application that provides a graphical interface for:

- Entering ODE functions with syntax highlighting
- Selecting solvers and parameters
- Visualizing results
- Comparing multiple solvers
- Access to a comprehensive library of example ODEs
- Exploring numerical methods for integration and differentiation
- Applying practical numerical applications

Design choices:
- Interactive interface with real-time feedback
- Organized library of example ODEs for quick testing
- Visualization tools for solution trajectories and numerical results
- Custom styling for a polished user experience
- Tabbed interface for different functionality

## Key Design Patterns and Principles

### Separation of Concerns

The codebase clearly separates different responsibilities:
- Parsing and validation are separate from solving
- Solver algorithms are separate from user interfaces
- Each solver implements a specific numerical method
- Numerical methods are organized by category

### Consistent Interfaces

All solvers and numerical methods follow consistent interface patterns:
- Input parameters follow a standard format
- Output structures are consistent across similar methods
- Optional parameters have sensible defaults

### Error Handling

Robust error handling is implemented throughout:
- Parsing errors are reported with specific line/column information
- Validation errors indicate the specific issue with input
- Solver failures provide informative error messages
- Numerical methods include checks for edge cases and invalid inputs

### Extensibility

The modular design makes it easy to extend:
- Adding a new solver requires implementing a single function or class
- New parsing capabilities can be added to the parsing module
- New user interfaces can be built on top of the existing solver functions
- Additional numerical methods can be integrated into the appropriate modules

## Interactions Between Components

The typical flow of data through the system is:

1. User provides an ODE as a string through CLI or web app
2. The string is parsed by `function_parser.py` or `symbolic_parser.py` into a callable function
3. The function is validated by `input_validation.py`
4. The validated function is passed to the appropriate solver
5. The solver computes the solution and returns arrays of time points and solution values
6. The interface (CLI or web app) presents the results to the user

For higher-order ODEs, an additional step occurs:
1. The higher-order ODE is converted to a system of first-order ODEs by `symbolic_parser.py`
2. The resulting system is then parsed by `function_parser.py`

For numerical applications:
1. User selects a numerical method and provides inputs
2. The selected function from the numerical module is called
3. Results are computed and returned
4. The interface displays the results, often with visualizations

## Performance Considerations

Several design choices contribute to the performance of the library:

1. Using NumPy for vectorized operations
2. Pre-compiling expressions with SymPy
3. Adaptive step-size methods for efficient handling of stiff problems
4. Caching of parsed functions
5. Optimized Newton iteration for implicit methods
6. Efficient implementation of numerical algorithms

## Testing and Quality Assurance

The project includes a comprehensive test suite in the `tests/` directory:

- Unit tests for individual components (parsing, validation, solvers, numerical methods)
- Integration tests for end-to-end workflows
- Validation against known analytical solutions
- Performance benchmarks
- Stability tests for stiff problems

## Conclusion

MOSES-ODE is designed to be a robust, extensible library for numerical mathematics, with a focus on:

1. **Usability**: Both through programmatic APIs and user interfaces
2. **Reliability**: Through comprehensive validation and error handling
3. **Performance**: With optimized implementations of various numerical methods
4. **Extensibility**: With a modular, well-structured codebase

The architecture permits easy addition of new solvers, features, and interfaces while maintaining consistency and quality throughout the system. The addition of the numerical module expands its capabilities beyond just ODE solving, making it a more comprehensive mathematical toolkit. 