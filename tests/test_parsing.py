import pytest
import numpy as np
import sympy as sp
from moses_ode.parsing.function_parser import parse_function_string
from moses_ode.parsing.symbolic_parser import convert_to_first_order, sys_of_ode

class TestFunctionParser:
    """Test suite for function_parser.py"""

    def test_parse_scalar_ode(self):
        """Test parsing a simple scalar ODE"""
        func_str = "-2*y + sin(t)"
        f = parse_function_string(func_str)
        t_val, y_val = 0.5, 1.0
        expected = -2 * y_val + np.sin(t_val)
        result = f(t_val, y_val)
        assert np.isclose(result, expected)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_parse_system_ode(self):
        """Test parsing a system of ODEs"""
        func_str = "[y1, -9.81 - 0.1*y1]"  # Simple pendulum with damping
        f = parse_function_string(func_str)
        t_val, y_val = 0.0, np.array([0.0, 1.0])
        expected = np.array([1.0, -9.81 - 0.1])
        result = f(t_val, y_val)
        assert np.allclose(result, expected)
        assert result.shape == (2,)

    def test_parse_system_with_coupling(self):
        """Test parsing a system with coupled equations"""
        func_str = "[-y0 + 2*y1, -y1 + sin(t)*y0]"
        f = parse_function_string(func_str)
        t_val, y_val = np.pi/2, np.array([1.0, 2.0])
        expected = np.array([-1.0 + 2*2.0, -2.0 + np.sin(np.pi/2)*1.0])
        result = f(t_val, y_val)
        assert np.allclose(result, expected)

    def test_scalar_with_no_t_dependency(self):
        """Test parsing an autonomous ODE (no explicit t dependency)"""
        func_str = "y**2 - y"
        f = parse_function_string(func_str)
        t_val, y_val = 10.0, 2.0  # t should not matter
        expected = y_val**2 - y_val
        result = f(t_val, y_val)
        assert np.isclose(result, expected)

    def test_scalar_with_no_y_dependency(self):
        """Test parsing an ODE with no y dependency"""
        func_str = "sin(t)"
        f = parse_function_string(func_str)
        t_val, y_val = np.pi/4, 100.0  # y should not matter
        expected = np.sin(t_val)
        result = f(t_val, y_val)
        assert np.isclose(result, expected)

    def test_system_with_different_size_inputs(self):
        """Test system handling with different sizes of input"""
        func_str = "[y0 + y1, y0 - y1]"
        f = parse_function_string(func_str)
        # Test with exact size
        t_val, y_val = 0.0, np.array([1.0, 2.0])
        expected = np.array([3.0, -1.0])
        result = f(t_val, y_val)
        assert np.allclose(result, expected)
        
        # Test with larger y input than needed
        y_val_extra = np.array([1.0, 2.0, 3.0])
        result_extra = f(t_val, y_val_extra)
        assert np.allclose(result_extra, expected)

    def test_parse_with_numpy_functions(self):
        """Test parsing expressions with NumPy functions"""
        func_str = "np.exp(-t) * y"
        f = parse_function_string(func_str)
        t_val, y_val = 1.0, 2.0
        expected = np.exp(-t_val) * y_val
        result = f(t_val, y_val)
        assert np.isclose(result, expected)

    def test_invalid_syntax(self):
        """Test parsing with invalid syntax"""
        func_str = "y +"  # Incomplete expression
        with pytest.raises(ValueError):
            parse_function_string(func_str)

    def test_invalid_variable(self):
        """Test parsing with invalid variables"""
        func_str = "z - y"  # z is not defined
        with pytest.raises(ValueError):
            parse_function_string(func_str)

    def test_system_mismatched_brackets(self):
        """Test parsing system with mismatched brackets"""
        func_str = "[y0, y1"  # Missing closing bracket
        with pytest.raises(ValueError):
            parse_function_string(func_str)

    def test_scalar_input_to_system(self):
        """Test passing scalar input to a system ODE function"""
        func_str = "[y0, -y0]"
        f = parse_function_string(func_str)
        t_val, y_val = 0.0, 1.0  # Scalar instead of array
        # This should convert scalar to array internally
        result = f(t_val, y_val)
        assert result.shape == (2,)


class TestSymbolicParser:
    """Test suite for symbolic_parser.py"""

    def test_convert_to_first_order_second_order(self):
        """Test converting 2nd order ODE to first order system"""
        # Create a second-order ODE: y'' = -9.81
        t = sp.Symbol('t')
        y = sp.Function('y')(t)
        ode = sp.Eq(sp.diff(y, t, t), -9.81)
        
        system, vars = convert_to_first_order(ode, y, t)
        
        # Expected: [y1' = y2, y2' = -9.81]
        assert len(system) == 2
        assert len(vars) == 2
        
        # Check first equation: y1' = y2
        first_eq = system[0]
        assert first_eq.lhs == sp.Derivative(vars[0], t)
        assert first_eq.rhs == vars[1]
        
        # Check second equation: y2' = -9.81
        second_eq = system[1]
        assert second_eq.lhs == sp.Derivative(vars[1], t)
        assert second_eq.rhs == -9.81

    def test_convert_already_first_order(self):
        """Test converting an already first order ODE"""
        # Create a first-order ODE: y' = -2*y
        t = sp.Symbol('t')
        y = sp.Function('y')(t)
        ode = sp.Eq(sp.diff(y, t), -2*y)
        
        system, vars = convert_to_first_order(ode, y, t)
        
        # Should return the same equation
        assert len(system) == 1
        assert len(vars) == 1
        assert system[0] == ode

    def test_convert_third_order(self):
        """Test converting 3rd order ODE to first order system"""
        # Create a third-order ODE: y''' = -y'' - y' - y
        t = sp.Symbol('t')
        y = sp.Function('y')(t)
        ode = sp.Eq(sp.diff(y, t, t, t), -sp.diff(y, t, t) - sp.diff(y, t) - y)
        
        system, vars = convert_to_first_order(ode, y, t)
        
        # Expected: [y1' = y2, y2' = y3, y3' = -y3 - y2 - y1]
        assert len(system) == 3
        assert len(vars) == 3

    def test_convert_invalid_inputs(self):
        """Test convert_to_first_order with invalid inputs"""
        t = sp.Symbol('t')
        y = sp.Function('y')(t)
        
        # Test with invalid ODE type
        with pytest.raises(TypeError):
            convert_to_first_order("not an equation", y, t)
        
        # Test with invalid y type
        with pytest.raises(TypeError):
            convert_to_first_order(sp.Eq(sp.diff(y, t), -y), "not a function", t)
        
        # Test with invalid t type
        with pytest.raises(TypeError):
            convert_to_first_order(sp.Eq(sp.diff(y, t), -y), y, "not a symbol")

    def test_sys_of_ode_second_order_basic(self):
        """Test sys_of_ode for a basic second-order ODE"""
        ode_str = "D2y = -9.81"
        result = sys_of_ode(ode_str)
        expected = "[y1, -9.81]" # Expected format after fix
        assert result == expected

    def test_sys_of_ode_third_order_basic(self):
        """Test sys_of_ode for a basic third-order ODE"""
        ode_str = "D3y = -D2y - D1y - y"
        result = sys_of_ode(ode_str)
        expected = "[y1, y2, -y2 - y1 - y0]" # Expected format after fix
        assert result == expected

    def test_sys_of_ode_with_coefficients_basic(self):
        """Test sys_of_ode with basic coefficients"""
        ode_str = "D2y = -4*D1y - 5*y"
        result = sys_of_ode(ode_str)
        expected = "[y1, -4*y1 - 5*y0]" # Expected format after fix
        assert result == expected

    @pytest.mark.parametrize("ode_str, expected_output", [
        # Order 1
        ("D1y = -2*y", "[ -2*y0]"), # Note: Handling for order 1 needs review in sys_of_ode
        ("Dy = t", "[t]"), # D == D1
        
        # Order 2
        ("D2y = 10", "[y1, 10]"),
        ("D2y = -k*y", "[y1, -k*y0]"),
        ("D2y = sin(t)", "[y1, sin(t)]"),
        ("D2y = -5*D1y - 10*y + cos(t)", "[y1, -5*y1 - 10*y0 + cos(t)]"),
        ("D2y = y**2 - D1y", "[y1, y0**2 - y1]"),
        ("D2y = exp(-t)*D1y", "[y1, exp(-t)*y1]"),
        (" D2y =  - 1 * Dy - 2 * y ", "[y1, - 1 * y1 - 2 * y0]"), # Extra spacing

        # Order 3
        ("D3y = 0", "[y1, y2, 0]"),
        ("D3y = D2y - D1y + y", "[y1, y2, y2 - y1 + y0]"),
        ("D3y = -D1y", "[y1, y2, -y1]"),
        ("D3y = t*D2y - y", "[y1, y2, t*y2 - y0]"),

        # Order 4
        ("D4y = y", "[y1, y2, y3, y0]"),
        ("D4y = -D3y - D2y - D1y - y", "[y1, y2, y3, -y3 - y2 - y1 - y0]"),
        
        # Higher Order (Example)
        ("D5y = D1y", "[y1, y2, y3, y4, y1]"),

        # Edge cases / Different formatting
        ("D2y= -y", "[y1, -y0]"), # No spaces around =
        ("D2y = - Dy", "[y1, - y1]"), # Using Dy instead of D1y
        ("D3y = - D2y", "[y1, y2, - y2]"),
    ])
    def test_sys_of_ode_parameterized(self, ode_str, expected_output):
        """Test sys_of_ode with a variety of D-notation inputs."""
        result = sys_of_ode(ode_str)
        # Normalize whitespace for comparison
        normalized_result = '[' + ', '.join(item.strip() for item in result.strip('[]').split(',')) + ']'
        normalized_expected = '[' + ', '.join(item.strip() for item in expected_output.strip('[]').split(',')) + ']'
        assert normalized_result == normalized_expected

    def test_sys_of_ode_order1_handling(self):
        """Check how sys_of_ode handles explicit 1st order."""
        # Current implementation might assume order >= 2 from D notation
        ode_str = "D1y = -k*y"
        result = sys_of_ode(ode_str)
        # The function logic starts result = "y1", but for order 1, there's no y1.
        # It replaces D1 with y1. This seems inconsistent.
        # Let's expect the direct RHS translation for order 1.
        expected = "[-k*y0]" 
        assert result == expected
        
        ode_str_dy = "Dy = -k*y" # uses Dy
        result_dy = sys_of_ode(ode_str_dy)
        assert result_dy == expected # Should handle Dy as D1y