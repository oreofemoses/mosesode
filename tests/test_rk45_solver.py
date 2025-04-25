import numpy as np
import pytest
from scipy.integrate import solve_ivp
from moses_ode.solvers.rk45 import rk45_solver

# Test Case 1: Simple Linear ODE dy/dt = -y, y(0) = 1
def test_linear_ode():
    def f(t, y): return -y
    t0, y0, t_end = 0, 1, 5
    h_init = 0.1

    t_vals, y_vals = rk45_solver(f, t0, y0, h_init, t_end)
    sol = solve_ivp(f, (t0, t_end), [y0], method='RK45', atol=1e-6, rtol=1e-6, dense_output=True)

    # Interpolate the scipy solution at the time points from our solver
    y_interp = sol.sol(t_vals)
    np.testing.assert_allclose(y_vals, y_interp[0], rtol=1e-3, atol=1e-3)

# Test Case 2: Nonlinear ODE dy/dt = y^2 - t, y(0) = 0
def test_nonlinear_ode():
    def f(t, y): return y**2 - t
    t0, y0, t_end = 0, 0, 2
    h_init = 0.1

    t_vals, y_vals = rk45_solver(f, t0, y0, h_init, t_end)
    sol = solve_ivp(f, (t0, t_end), [y0], method='RK45', atol=1e-6, rtol=1e-6,dense_output=True)

    # Interpolate the scipy solution at the time points from our solver
    y_interp = sol.sol(t_vals)

    np.testing.assert_allclose(y_vals, y_interp[0], rtol=1e-3, atol=1e-3)

# Test Case 3: Edge Case - Tiny Step Size
def test_tiny_step_size():
    def f(t, y): return -y
    t0, y0, t_end = 0, 1, 5
    h_init = 1e-8  # Very small step size

    t_vals, y_vals = rk45_solver(f, t0, y0, h_init, t_end)
    assert len(t_vals) > 10  # Ensure it progresses
    assert t_vals[-1] == pytest.approx(t_end, rel=1e-3)

# Test Case 4: Large Integration Range
def test_large_integration():
    def f(t, y): return -0.5 * y
    t0, y0, t_end = 0, 10, 1000  # Large range
    h_init = 1.0

    t_vals, y_vals = rk45_solver(f, t0, y0, h_init, t_end)
    assert len(t_vals) < 5000  # Ensure it adapts step sizes efficiently
    assert t_vals[-1] == pytest.approx(t_end, rel=1e-3)

# Test Case 5: Stiff ODE dy/dt = -1000(y - cos(t)), y(0) = 1
def test_stiff_ode():
    def f(t, y): return -1000 * (y - np.cos(t))
    t0, y0, t_end = 0, 1, 1
    h_init = 0.01

    t_vals, y_vals = rk45_solver(f, t0, y0, h_init, t_end)
    assert len(t_vals) > 10  # Should take many steps due to stiffness


def test_system_of_odes():
    def f(t, y): return np.array([y[1], -y[0]])  # dy1/dt = y2, dy2/dt = -y1

    t0, y0, t_end = 0, np.array([1, 0]), 10  # Ensure y0 is also a NumPy array
    h_init = 0.1

    t_vals, y_vals = rk45_solver(f, t0, y0, h_init, t_end)
    sol = solve_ivp(f, (t0, t_end), y0, method='RK45', atol=1e-6, rtol=1e-6, dense_output=True)

    y_interp = sol.sol(t_vals)  # Shape: (2, N)

    # Transpose to match (N, 2) shape of y_vals
    np.testing.assert_allclose(y_vals, y_interp.T, rtol=1e-3, atol=1e-3)
