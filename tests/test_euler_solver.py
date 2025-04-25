import pytest
import numpy as np
from moses_ode.solvers.euler import euler_solver

def test_euler_solver_scalar():
    """
    Test Euler solver with a simple scalar ODE dy/dt = -2y.
    The Euler method is a first-order method, so we use a relaxed tolerance.
    """
    def f(t, y):
        return -2 * y

    t0, y0, h, t_end = 0, 1.0, 0.01, 1.0
    t_values, y_values = euler_solver(f, t0, y0, h, t_end)

    # Expected solution: y(t) = exp(-2t)
    y_exact = np.exp(-2 * t_values)
    np.testing.assert_allclose(y_values, y_exact, rtol=0.2)  # Relaxed tolerance

def test_euler_solver_vector():
    """
    Test Euler solver with a system of ODEs.
    The Euler method is a first-order method, so we use a relaxed tolerance.
    """
    def f(t, y):
        return np.array([-2 * y[0], -3 * y[1]])

    t0, y0, h, t_end = 0, np.array([1.0, 2.0]), 0.01, 1.0
    t_values, y_values = euler_solver(f, t0, y0, h, t_end)

    # Expected solutions: y1(t) = exp(-2t), y2(t) = 2 * exp(-3t)
    y_exact = np.array([np.exp(-2 * t_values), 2 * np.exp(-3 * t_values)]).T
    np.testing.assert_allclose(y_values, y_exact, rtol=0.2)  # Relaxed tolerance

def test_euler_solver_invalid_step():
    """Test Euler solver with invalid step size."""
    def f(t, y):
        return -2 * y

    with pytest.raises(ValueError, match="Step size h must be positive"):
        euler_solver(f, 0, 1.0, -0.1, 1.0)

def test_euler_solver_invalid_time():
    """Test Euler solver with t_end <= t0."""
    def f(t, y):
        return -2 * y

    with pytest.raises(ValueError, match="End time t_end must be greater than initial time t0"):
        euler_solver(f, 1.0, 1.0, 0.1, 0.5)

def test_euler_solver_zero_step():
    """Test Euler solver when step size is zero."""
    def f(t, y):
        return -2 * y

    with pytest.raises(ValueError, match="Step size h must be positive"):
        euler_solver(f, 0, 1.0, 0, 1.0)

def test_euler_solver_single_step():
    """Test Euler solver for a single step integration."""
    def f(t, y):
        return -2 * y

    t0, y0, h, t_end = 0, 1.0, 1.0, 1.0  # Single step
    t_values, y_values = euler_solver(f, t0, y0, h, t_end)

    assert len(t_values) == 2
    assert len(y_values) == 2
    np.testing.assert_allclose(y_values[1], y0 + h * f(t0, y0), rtol=1e-5)

def test_euler_solver_large_step():
    """
    Test Euler solver with a large step size.
    The Euler method performs poorly with large step sizes, so we use a very relaxed tolerance.
    """
    def f(t, y):
        return -2 * y

    t0, y0, h, t_end = 0, 1.0, 0.05, 1.0
    t_values, y_values = euler_solver(f, t0, y0, h, t_end)

    # Expected solution: y(t) = exp(-2t)
    y_exact = np.exp(-2 * t_values)
    np.testing.assert_allclose(y_values, y_exact, rtol=0.5)  # Very relaxed tolerance

def test_euler_solver_nonlinear():
    """
    Test Euler solver with a non-linear ODE dy/dt = -y^2.
    The Euler method is a first-order method, so we use a relaxed tolerance.
    """
    def f(t, y):
        return -y**2

    t0, y0, h, t_end = 0, 1.0, 0.1, 1.0
    t_values, y_values = euler_solver(f, t0, y0, h, t_end)

    # Expected solution: y(t) = 1 / (t + 1)
    y_exact = 1 / (t_values + 1)
    np.testing.assert_allclose(y_values, y_exact, rtol=0.2)  # Relaxed tolerance

if __name__ == "__main__":
    pytest.main()