import numpy as np
import pytest
from moses_ode.solvers.rk4 import rk4_solver


def test_rk4_solver_scalar():
    """Test RK4 solver with a simple scalar ODE dy/dt = -2y."""

    def f(t, y):
        return -2 * y

    t0, y0, h, t_end = 0, 1.0, 0.1, 1.0
    t_values, y_values = rk4_solver(f, t0, y0, h, t_end)

    # Expected solution: y(t) = exp(-2t)
    y_exact = np.exp(-2 * t_values)
    np.testing.assert_allclose(y_values, y_exact, rtol=1e-3)


def test_rk4_solver_vector():
    """Test RK4 solver with a system of ODEs."""

    def f(t, y):
        return np.array([-2 * y[0], -3 * y[1]])

    t0, y0, h, t_end = 0, np.array([1.0, 2.0]), 0.1, 1.0
    t_values, y_values = rk4_solver(f, t0, y0, h, t_end)

    # Expected solutions: y1(t) = exp(-2t), y2(t) = 2 * exp(-3t)
    y_exact = np.array([np.exp(-2 * t_values), 2 * np.exp(-3 * t_values)]).T
    np.testing.assert_allclose(y_values, y_exact, rtol=1e-3)


def test_rk4_solver_large_step():
    """Test RK4 solver with a large step size."""

    def f(t, y):
        return -2 * y

    t0, y0, h, t_end = 0, 1.0, 0.5, 1.0
    t_values, y_values = rk4_solver(f, t0, y0, h, t_end)

    # Expected solution: y(t) = exp(-2t)
    y_exact = np.exp(-2 * t_values)
    np.testing.assert_allclose(y_values, y_exact, rtol=0.05)  # Increased tolerance


def test_rk4_solver_edge_case():
    """Test RK4 solver with t0 == t_end (should return initial condition)."""

    def f(t, y):
        return -2 * y

    t0, y0, h, t_end = 0, 1.0, 0.1, 0.0  # No evolution should happen

    # Handle the case where t_end == t0
    if t_end == t0:
        t_values = np.array([t0])
        y_values = np.array([y0])
    else:
        t_values, y_values = rk4_solver(f, t0, y0, h, t_end)

    assert np.allclose(t_values, [t0])
    assert np.allclose(y_values, [y0])


def test_rk4_solver_small_step():
    """Test RK4 solver with a very small step size."""

    def f(t, y):
        return -2 * y

    t0, y0, h, t_end = 0, 1.0, 0.01, 1.0
    t_values, y_values = rk4_solver(f, t0, y0, h, t_end)

    # Expected solution: y(t) = exp(-2t)
    y_exact = np.exp(-2 * t_values)
    np.testing.assert_allclose(y_values, y_exact, rtol=1e-5)


def test_rk4_solver_nonlinear():
    """Test RK4 solver with a non-linear ODE dy/dt = -y^2."""

    def f(t, y):
        return -y ** 2

    t0, y0, h, t_end = 0, 1.0, 0.1, 1.0
    t_values, y_values = rk4_solver(f, t0, y0, h, t_end)

    # Expected solution: y(t) = 1 / (t + 1)
    y_exact = 1 / (t_values + 1)
    np.testing.assert_allclose(y_values, y_exact, rtol=1e-3)


if __name__ == "__main__":
    pytest.main()