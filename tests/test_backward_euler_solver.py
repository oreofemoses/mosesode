import numpy as np
import pytest
from scipy.integrate import solve_ivp
from moses_ode.solvers.backward_euler import backward_euler_solver

# Test 1: Simple Linear ODE dy/dt = -y (Exponential Decay)
def test_linear_ode():
    def f(t, y): return -y

    t0, y0, t_end, h = 0, 1, 2, 0.001
    t_vals, y_vals = backward_euler_solver(f, t0, y0, h, t_end)

    # Compare with analytical solution: y = exp(-t)
    y_exact = np.exp(-t_vals)
    np.testing.assert_allclose(y_vals, y_exact, rtol=5e-2, atol=5e-2)

# Test 2: Nonlinear ODE dy/dt = y^2 - y
def test_nonlinear_ode():
    def f(t, y): return y**2 - y

    t0, y0, t_end, h = 0, 0.5, 2, 0.1
    t_vals, y_vals = backward_euler_solver(f, t0, y0, h, t_end)

    # Compare with SciPy's solve_ivp solution
    sol = solve_ivp(f, (t0, t_end), [y0], method='BDF', atol=1e-6, rtol=1e-6, dense_output=True)
    y_interp = sol.sol(t_vals).flatten()

    np.testing.assert_allclose(y_vals, y_interp, rtol=5e-2, atol=5e-2)

# Test 3: System of ODEs (Harmonic Oscillator)
def test_system_of_odes():
    def f(t, y): return np.array([y[1], -y[0]])  # dy1/dt = y2, dy2/dt = -y1

    t0, y0, t_end, h = 0, [1, 0], 10, 0.001
    t_vals, y_vals = backward_euler_solver(f, t0, y0, h, t_end)

    # Compare with SciPy's solve_ivp
    sol = solve_ivp(f, (t0, t_end), y0, method='BDF', atol=1e-6, rtol=1e-6, dense_output=True)
    y_interp = np.atleast_2d(sol.sol(t_vals)).T  # Ensure shape compatibility

    np.testing.assert_allclose(y_vals, y_interp, rtol=5e-2, atol=5e-2)

# Test 4: Edge Case - Zero Initial Condition
def test_zero_initial_condition():
    def f(t, y): return -2 * y

    t0, y0, t_end, h = 0, 0, 1, 0.1
    t_vals, y_vals = backward_euler_solver(f, t0, y0, h, t_end)

    np.testing.assert_allclose(y_vals, np.zeros_like(y_vals), atol=1e-5)

# Test 5: Stiff ODE dy/dt = -1000 * y
def test_stiff_ode():
    def f(t, y): return -1000 * y
    t0, y0, t_end, h = 0, 1, 0.1, 0.00001
    t_vals, y_vals = backward_euler_solver(f, t0, y0, h, t_end)

    # Compare with analytical solution: y = exp(-1000*t)
    y_exact = np.exp(-1000*t_vals)
    np.testing.assert_allclose(y_vals, y_exact, rtol=5e-2, atol=5e-2)

# Test 6: User-Supplied Jacobian
def test_user_supplied_jacobian():
    def f(t, y): return -y**3
    def jac(t, y): return -3*y**2  # Analytical Jacobian

    t0, y0, t_end, h = 0, 1, 1, 0.1
    t_vals_numerical, y_vals_numerical = backward_euler_solver(f, t0, y0, h, t_end)
    t_vals_analytical, y_vals_analytical = backward_euler_solver(f, t0, y0, h, t_end, jac=jac)

    np.testing.assert_allclose(y_vals_numerical, y_vals_analytical, rtol=5e-3, atol=5e-3)

#Test 7: Step Size Reduction
def test_step_size_reduction():
    def f(t, y): return -y**2 #y' = y^2, which has a singularity at t = 1 if y(0) = 1
    t0, y0, t_end, h = 0, 1, 0.9, 0.2 #h is too large and will cause Newton to fail initially
    t_vals, y_vals = backward_euler_solver(f, t0, y0, h, t_end)
    assert h > (t_end - t0)/(len(t_vals)-1) #Check if step size has been reduced


if __name__ == "__main__":
    pytest.main()
