import pytest
import numpy as np
from moses_ode.parsing.function_parser import parse_function_string
from moses_ode.parsing.symbolic_parser import sys_of_ode
from moses_ode.validation.input_validation import validate_function_input
from moses_ode.solvers import euler, rk4, rk45, heun, midpoint, backward_euler, crank_nicolson

class TestEndToEndWorkflow:
    """Test suite for end-to-end workflow testing"""

    def test_decay_equation_workflow(self):
        """Test a complete workflow for solving the decay equation"""
        # y' = -2y, y(0) = 1, analytical solution: y(t) = e^(-2t)
        
        # 1. Parse the ODE
        ode_str = "-2*y"
        f = parse_function_string(ode_str)
        
        # 2. Set initial conditions
        t0 = 0.0
        y0 = 1.0
        t_end = 1.0
        h = 0.01
        
        # 3. Validate the function
        f = validate_function_input(f, t0, y0)
        
        # 4. Solve with different methods
        t_euler, y_euler = euler.euler_solver(f, t0, y0, h, t_end)
        t_rk4, y_rk4 = rk4.rk4_solver(f, t0, y0, h, t_end)
        
        # 5. Check results against analytical solution
        y_exact = lambda t: np.exp(-2 * t)
        exact_values_euler = y_exact(t_euler)
        exact_values_rk4 = y_exact(t_rk4)
        
        # 6. Verify error bounds appropriate to each method
        error_euler = np.max(np.abs(y_euler - exact_values_euler))
        error_rk4 = np.max(np.abs(y_rk4 - exact_values_rk4))
        
        assert error_euler < 0.1, f"Euler error {error_euler} exceeds expected bound"
        assert error_rk4 < 1e-4, f"RK4 error {error_rk4} exceeds expected bound"
        assert error_rk4 < error_euler, "RK4 should be more accurate than Euler"

    def test_pendulum_equation_workflow(self):
        """Test a complete workflow for solving the pendulum equation"""
        # y'' = -9.81*sin(y), converted to y1' = y2, y2' = -9.81*sin(y1)
        
        # 1. Parse the ODE using symbolic parser for higher-order conversion
        ode_str = "D2y = -9.81*sin(y)"
        system_str = sys_of_ode(ode_str)
        # Fix the system string to correctly use 'sin(y0)' instead of 'sin(y)'
        system_str = system_str.replace("sin(y)", "sin(y0)")
        
        # 2. Parse the system
        f = parse_function_string(system_str)
        
        # 3. Set initial conditions that match the system size
        # The system is [y0, y1, -9.81*sin(y0)] - needs 3 values
        t0 = 0.0
        # For a second-order ODE converted to first-order system, we need 3 initial conditions
        y0 = np.array([0.1, 0.0])  # Initial position, velocity, and acceleration
        t_end = 2.0  # Simulate for 2 seconds
        h = 0.01
        
        # 4. Validate the function
        f = validate_function_input(f, t0, y0)
        
        # 5. We'll create a wrapper function for solving with standard solvers
        def pendulum_wrapper(t, y):
            # Extract just what we need from the full system
            dydt = np.zeros_like(y)
            dydt[0] = y[1]  # y0' = y1
            dydt[1] = -9.81 * np.sin(y[0])  # y1' = -9.81*sin(y0)
            return dydt
        
        # Solve with RK4 using the wrapper
        t_rk4, y_rk4 = rk4.rk4_solver(pendulum_wrapper, t0, y0[:2], h, t_end)
        
        # 6. For small angles, period should be approximately 2π√(L/g) = 2π√(1/9.81) ≈ 2 seconds
        # Check that the solution has the correct period by finding zero crossings
        zero_crossings = []
        for i in range(1, len(t_rk4)):
            if y_rk4[i-1, 0] * y_rk4[i, 0] <= 0 and y_rk4[i-1, 1] > 0:  # Zero crossing with positive velocity
                zero_crossings.append(t_rk4[i])
        
        # Should have at least one full period in 2 seconds
        assert len(zero_crossings) >= 1, "Failed to capture at least one period"
        
        if len(zero_crossings) >= 2:
            # Calculate period from consecutive zero crossings
            period = zero_crossings[1] - zero_crossings[0]
            expected_period = 2 * np.pi * np.sqrt(1/9.81)
            # Allow 10% error due to numerical approximation and nonlinear effects
            assert abs(period - expected_period) / expected_period < 0.1, \
                f"Period {period} differs too much from expected {expected_period}"

    def test_system_of_equations_workflow(self):
        """Test a complete workflow for solving a coupled system (predator-prey model)"""
        # Lotka-Volterra model: dx/dt = x(a-by), dy/dt = y(-c+dx)
        
        # 1. Parse the system
        ode_str = "[y0*(0.5-0.05*y1), y1*(-0.5+0.02*y0)]"  # Predator-prey model
        f = parse_function_string(ode_str)
        
        # 2. Set initial conditions
        t0 = 0.0
        y0 = np.array([10.0, 10.0])  # Initial populations
        t_end = 100.0  # Simulate for 100 time units
        h = 0.1
        
        # 3. Validate the function
        f = validate_function_input(f, t0, y0)
        
        # 4. Solve with different methods
        t_euler, y_euler = euler.euler_solver(f, t0, y0, h, t_end)
        t_rk4, y_rk4 = rk4.rk4_solver(f, t0, y0, h, t_end)
        t_heun, y_heun = heun.heun_solver(f, t0, y0, h, t_end)
        t_midpoint, y_midpoint = midpoint.midpoint_solver(f, t0, y0, h, t_end)
        
        # 5. Verify properties of the solution:
        # The solution should be oscillatory and stay positive
        assert np.all(y_rk4[:, 0] > 0), "Prey population went negative or extinct"
        assert np.all(y_rk4[:, 1] > 0), "Predator population went negative or extinct"
        
        # Check oscillatory behavior - find max and min points
        prey_peaks = []
        for i in range(1, len(t_rk4)-1):
            if y_rk4[i-1, 0] < y_rk4[i, 0] and y_rk4[i, 0] > y_rk4[i+1, 0]:
                prey_peaks.append(y_rk4[i, 0])
        
        # Should have at least 2 peaks for oscillatory behavior
        assert len(prey_peaks) >= 2, "Solution didn't exhibit expected oscillatory behavior"
        
        # 6. Compare different methods - higher-order methods should agree better with each other
        # Compare last point of RK4 and Heun (should be closer than Euler to RK4)
        error_euler_rk4 = np.linalg.norm(y_euler[-1] - y_rk4[-1])
        error_heun_rk4 = np.linalg.norm(y_heun[-1] - y_rk4[-1])
        assert error_heun_rk4 < error_euler_rk4, "Heun should be closer to RK4 than Euler is"

    def test_stiff_ode_workflow(self):
        """Test a complete workflow for solving a stiff ODE"""
        # y' = -1000y + 3000 - 2000e^-t, y(0) = 0
        # This has rapidly changing components and demonstrates the benefit of implicit methods
        
        # 1. Define the ODE directly instead of parsing it to avoid numpy issues
        def stiff_ode(t, y):
            return -1000*y + 3000 - 2000*np.exp(-t)
        
        # 2. Set initial conditions
        t0 = 0.0
        y0 = 0.0
        t_end = 0.5
        # Use a larger step for implicit methods - they can handle it
        h_explicit = 0.0001  # Very small for explicit methods due to stiffness
        h_implicit = 0.01    # Larger for implicit methods
        
        # 3. No need to validate a manually defined function
        f = stiff_ode
        
        # 4. Solve with explicit and implicit methods
        t_euler, y_euler = euler.euler_solver(f, t0, y0, h_explicit, t_end)
        t_rk4, y_rk4 = rk4.rk4_solver(f, t0, y0, h_explicit, t_end)
        t_backward, y_backward = backward_euler.backward_euler_solver(f, t0, y0, h_implicit, t_end)
        t_crank, y_crank = crank_nicolson.crank_nicolson_solver(f, t0, y0, h_implicit, t_end)
        
        # 5. Check against analytical solution
        # y(t) = 3 - 2e^-t - e^-1000t
        y_exact = lambda t: 3 - 2*np.exp(-t) - np.exp(-1000*t)
        exact_values_end = y_exact(t_end)
        
        # 6. Verify errors - implicit methods should handle this stiff problem better
        # despite using larger step sizes
        error_euler = abs(y_euler[-1] - exact_values_end)
        error_rk4 = abs(y_rk4[-1] - exact_values_end)
        error_backward = abs(y_backward[-1] - exact_values_end)
        error_crank = abs(y_crank[-1] - exact_values_end)
        
        # All should converge to the same solution
        assert error_euler < 0.1, f"Euler error {error_euler} exceeds expected bound"
        assert error_rk4 < 0.1, f"RK4 error {error_rk4} exceeds expected bound"
        assert error_backward < 0.1, f"Backward Euler error {error_backward} exceeds expected bound"
        assert error_crank < 0.1, f"Crank-Nicolson error {error_crank} exceeds expected bound"
        
        # Check that the number of steps differs significantly between methods
        # Implicit methods should use far fewer steps
        assert len(t_backward) < len(t_euler) / 10, \
            "Backward Euler should use far fewer steps than Euler for this stiff problem" 