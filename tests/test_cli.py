import pytest
from click.testing import CliRunner
import numpy as np
import os
from cli import cli, solve, benchmark

class TestCLI:
    """Test suite for CLI functionality"""

    def setup_method(self):
        """Setup method that runs before each test"""
        self.runner = CliRunner()

    def test_main_help(self):
        """Test main CLI help command"""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "MOSES-ODE: Numerical ODE Solver (CLI)" in result.output
        assert "solve" in result.output
        assert "benchmark" in result.output

    def test_solve_help(self):
        """Test solve command help"""
        result = self.runner.invoke(cli, ["solve", "--help"])
        assert result.exit_code == 0
        assert "Solve an ODE with a specified numerical method." in result.output
        assert "--ode" in result.output
        assert "--solver" in result.output
        assert "--t0" in result.output
        assert "--y0" in result.output
        assert "--t_end" in result.output

    def test_benchmark_help(self):
        """Test benchmark command help"""
        result = self.runner.invoke(cli, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "Benchmark MOSES-ODE solvers on a predefined problem." in result.output
        assert "--problem" in result.output
        assert "--solvers" in result.output
        assert "--t_end" in result.output

    def test_solve_first_order_ode(self):
        """Test solving a first-order ODE with the CLI"""
        result = self.runner.invoke(cli, [
            "solve",
            "--ode", "-2*y",
            "--solver", "euler",
            "--y0", "1.0",
            "--t_end", "1.0",
            "--h", "0.1"
        ])
        assert result.exit_code == 0
        assert "Solution Summary" in result.output
        # Check that the table has proper headers
        assert "Time (t)" in result.output
        assert "y(t)" in result.output

    def test_solve_system_ode(self):
        """Test solving a system of ODEs with the CLI"""
        result = self.runner.invoke(cli, [
            "solve",
            "--ode", "[y1, -9.81]",  # Simple pendulum (second-order converted to system)
            "--solver", "rk4",
            "--y0", "[0, 0]",
            "--t_end", "1.0",
            "--h", "0.1"
        ])
        assert result.exit_code == 0
        assert "Solution Summary" in result.output
        # Check for system output headers
        assert "Time (t)" in result.output
        assert "y_0(t)" in result.output # Check for first component of system output
        assert "y_1(t)" in result.output # Check for second component of system output

    def test_solve_higher_order_ode(self):
        """Test solving a higher-order ODE with the CLI (using D notation)"""
        result = self.runner.invoke(cli, [
            "solve",
            "--ode", "D2y = -9.81",  # Simple pendulum (second-order)
            "--solver", "rk45",
            "--y0", "[0, 0]",  # We need 3 initial conditions for this system
            "--t_end", "1.0"
        ])
        assert result.exit_code == 0
        assert "sys_of_ode output" in result.output
        assert "Solution Summary" in result.output

    def test_solve_with_output_file(self, tmpdir):
        """Test solving and saving output to a file"""
        output_file = os.path.join(tmpdir, "output.csv")
        result = self.runner.invoke(cli, [
            "solve",
            "--ode", "-2*y",
            "--solver", "euler",
            "--y0", "1.0",
            "--t_end", "1.0",
            "--h", "0.1",
            "--output", output_file
        ])
        assert result.exit_code == 0
        assert os.path.exists(output_file)
        data = np.loadtxt(output_file, delimiter=",", skiprows=1)
        assert data.shape[0] > 0  # Should have some rows of data
        assert data.shape[1] == 2  # t and y columns

    def test_solve_missing_required_params(self):
        """Test error handling for missing required parameters"""
        # Missing --ode parameter
        result = self.runner.invoke(cli, [
            "solve",
            "--solver", "euler",
            "--y0", "1.0",
            "--t_end", "1.0",
            "--h", "0.1"
        ])
        assert result.exit_code != 0

        # Missing --y0 parameter
        result = self.runner.invoke(cli, [
            "solve",
            "--ode", "-2*y",
            "--solver", "euler",
            "--t_end", "1.0",
            "--h", "0.1"
        ])
        assert result.exit_code != 0

    def test_solve_fixed_step_without_h(self):
        """Test error when using fixed-step solver without specifying step size"""
        result = self.runner.invoke(cli, [
            "solve",
            "--ode", "-2*y",
            "--solver", "euler",
            "--y0", "1.0",
            "--t_end", "1.0"
        ])
        assert "requires step size 'h'" in result.output

    def test_benchmark_command(self):
        """Test the benchmark command"""
        result = self.runner.invoke(cli, [
            "benchmark",
            "--problem", "decay",
            "--solvers", "euler,rk4",
            "--t_end", "1.0"
        ])
        assert result.exit_code == 0
        assert "Benchmark Results" in result.output
        assert "Solver" in result.output
        assert "Steps" in result.output
        assert "Max Error" in result.output
        assert "euler" in result.output
        assert "rk4" in result.output

    def test_benchmark_invalid_solver(self):
        """Test benchmark with invalid solver"""
        result = self.runner.invoke(cli, [
            "benchmark",
            "--problem", "decay",
            "--solvers", "invalid_solver",
            "--t_end", "1.0"
        ])
        assert "Invalid solver" in result.output

    def test_benchmark_invalid_problem(self):
        """Test benchmark with invalid problem"""
        result = self.runner.invoke(cli, [
            "benchmark",
            "--problem", "invalid_problem",
            "--solvers", "euler",
            "--t_end", "1.0"
        ])
        assert result.exit_code != 0  # Should fail with invalid problem 