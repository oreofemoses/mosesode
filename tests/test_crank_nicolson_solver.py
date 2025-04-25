import numpy as np
import unittest
from moses_ode.solvers.crank_nicolson import crank_nicolson_solver
from scipy.integrate import solve_ivp  # For comparison

class TestCrankNicolsonSolver(unittest.TestCase):
    def test_linear_ode(self):
        # dy/dt = -y, solution: y = y0 * exp(-t)
        def f(t, y):
            return -y

        t0, y0, h, t_end = 0, 1, 0.1, 1.0
        t_vals, y_vals = crank_nicolson_solver(f, t0, y0, h, t_end)

        y_exact = np.exp(-np.array(t_vals))
        np.testing.assert_allclose(y_vals, y_exact, rtol=5e-2, atol=5e-2)

    def test_non_linear_ode(self):
        # dy/dt = -y^2, solution: y = 1 / (1 + t)
        def f(t, y):
            return -y ** 2

        t0, y0, h, t_end = 0, 1, 0.1, 1.0
        t_vals, y_vals = crank_nicolson_solver(f, t0, y0, h, t_end)

        y_exact = 1 / (1 + np.array(t_vals))
        np.testing.assert_allclose(y_vals, y_exact, rtol=5e-2, atol=5e-2)

        # Comparison with SciPy's solve_ivp
        sol = solve_ivp(f, (t0, t_end), [y0], method='RK45', dense_output=True)
        y_interp = sol.sol(t_vals).flatten()
        np.testing.assert_allclose(y_vals, y_interp, rtol=5e-2, atol=5e-2)

    def test_system_of_odes(self):
        # System: dx/dt = -y, dy/dt = x, circular motion
        def f(t, y):
            return np.array([-y[1], y[0]])

        t0, y0, h, t_end = 0, np.array([1, 0]), 0.1, 1.0
        t_vals, y_vals = crank_nicolson_solver(f, t0, y0, h, t_end)

        y_exact_0 = np.cos(np.array(t_vals))
        y_exact_1 = np.sin(np.array(t_vals))
        np.testing.assert_allclose(y_vals[:, 0], y_exact_0, rtol=5e-2, atol=5e-2)
        np.testing.assert_allclose(y_vals[:, 1], y_exact_1, rtol=5e-2, atol=5e-2)

        #Magnitude should be approximately constant (circular motion)
        np.testing.assert_allclose(np.linalg.norm(y_vals, axis=1), np.linalg.norm(y0), rtol=5e-2, atol=5e-2)

        # Comparison with SciPy's solve_ivp
        sol = solve_ivp(f, (t0, t_end), y0, method='RK45', dense_output=True)
        y_interp = np.atleast_2d(sol.sol(t_vals)).T
        np.testing.assert_allclose(y_vals, y_interp, rtol=5e-2, atol=5e-2)


    def test_zero_initial_condition(self):
        def f(t, y): return -2 * y
        t0, y0, h, t_end = 0, 0, 1, 0.1
        t_vals, y_vals = crank_nicolson_solver(f, t0, y0, h, t_end)
        np.testing.assert_allclose(y_vals, np.zeros_like(y_vals), atol=1e-5, rtol=1e-5)

    def test_nonzero_initial_time(self):
        def f(t, y): return -y
        t0, y0, h, t_end = 1, 0.5, 0.1, 2.0  # t0 = 1
        t_vals, y_vals = crank_nicolson_solver(f, t0, y0, h, t_end)
        y_exact = 0.5 * np.exp(-np.array(t_vals - t0)) # Shifted exponential
        np.testing.assert_allclose(y_vals, y_exact, rtol=5e-2, atol=5e-2)

    def test_analytical_jacobian(self):
        def f(t, y): return -y**3
        def jac(t, y): return -3*y**2  # Analytical Jacobian
        t0, y0, h, t_end = 0, 1, 0.1, 1.0
        t_vals_num_jac, y_vals_num_jac = crank_nicolson_solver(f, t0, y0, h, t_end) # Numerical Jacobian
        t_vals_analytic_jac, y_vals_analytic_jac = crank_nicolson_solver(f, t0, y0, h, t_end, jac=jac) #User-supplied Jacobian
        np.testing.assert_allclose(y_vals_num_jac, y_vals_analytic_jac, rtol=5e-3, atol=5e-3) #Compare solutions


    def test_stiff_ode(self):
        def f(t, y): return -100 * y # Moderately stiff
        t0, y0, h, t_end = 0, 1, 0.01, 0.5 # Smaller step size for stiff problem
        t_vals, y_vals = crank_nicolson_solver(f, t0, y0, h, t_end)
        y_exact = np.exp(-100 * np.array(t_vals))
        np.testing.assert_allclose(y_vals, y_exact, rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    unittest.main()