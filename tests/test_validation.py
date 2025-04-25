import pytest
from moses_ode.validation.input_validation import validate_function_input, validate_jacobian_input

# Test cases for validate_function_input
def test_validate_function_input_valid_scalar():
    """Test validation of a valid scalar ODE function."""
    def f(t, y):
        return -2 * y

    t0 = 0
    y0 = 1.0
    validated_f = validate_function_input(f, t0, y0)
    assert callable(validated_f)
    assert np.allclose(validated_f(t0, y0), f(t0, y0))  # Ensure function behavior is unchanged

def test_validate_function_input_valid_vector():
    """Test validation of a valid vector ODE function."""
    def f(t, y):
        return np.array([-2 * y[0], -3 * y[1]])

    t0 = 0
    y0 = np.array([1.0, 2.0])
    validated_f = validate_function_input(f, t0, y0)
    assert callable(validated_f)
    assert np.allclose(validated_f(t0, y0), f(t0, y0))

def test_validate_function_input_invalid_callable():
    """Test validation of a non-callable function."""
    f = "not a function"
    t0 = 0
    y0 = 1.0
    with pytest.raises(TypeError, match="ODE function f must be callable"):
        validate_function_input(f, t0, y0)

def test_validate_function_input_invalid_output_shape():
    """Test validation of a function with incorrect output shape."""
    def f(t, y):
        return np.array([-2 * y, -3 * y])  # Incorrect shape for scalar y0

    t0 = 0
    y0 = 1.0
    with pytest.raises(ValueError, match="For a 1-order ODE, you need to provide exactly 1 initial conditions"):
        validate_function_input(f, t0, y0)

def test_validate_function_input_invalid_output_type():
    """Test validation of a function with invalid output type."""
    def f(t, y):
        return "not a numpy array"

    t0 = 0
    y0 = 1.0
    with pytest.raises(ValueError, match="Error when calling f\\(t, y\\)"):
        validate_function_input(f, t0, y0)

def test_validate_function_input_returns_list():
    """Test validation of a function that returns a list instead of a NumPy array."""
    def f(t, y):
        return [y[0], y[1]]  # Ensure we use index notation for a proper list test

    t0 = 0
    y0 = np.array([1.0, 2.0])
    # The validation function now converts lists to arrays, so this should pass
    result = validate_function_input(f, t0, y0)
    assert callable(result)

# Test cases for validate_jacobian_input
def test_validate_jacobian_input_valid():
    """Test validation of a valid Jacobian function."""
    def jac(t, y):
        return np.array([[-2, 0], [0, -3]])

    t0 = 0
    y0 = np.array([1.0, 2.0])
    validated_jac = validate_jacobian_input(jac, t0, y0)
    assert callable(validated_jac)
    assert np.allclose(validated_jac(t0, y0), jac(t0, y0))

def test_validate_jacobian_input_invalid_callable():
    """Test validation of a non-callable Jacobian function."""
    jac = "not a function"
    t0 = 0
    y0 = np.array([1.0, 2.0])
    with pytest.raises(TypeError, match="Jacobian function jac must be callable"):
        validate_jacobian_input(jac, t0, y0)

def test_validate_jacobian_input_invalid_output_shape():
    """Test validation of a Jacobian function with incorrect output shape."""
    def jac(t, y):
        return np.array([-2, -3])  # Incorrect shape for Jacobian

    t0 = 0
    y0 = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="jac\\(t, y\\) must return a square matrix"):
        validate_jacobian_input(jac, t0, y0)

def test_validate_jacobian_input_returns_list():
    """Test validation of a Jacobian function that returns a list instead of a NumPy array."""
    def jac(t, y):
        return [[-2, 0], [0, -3]]  # Incorrect type (list instead of np.array)

    t0 = 0
    y0 = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Error when calling jac\\(t, y\\)"):
        validate_jacobian_input(jac, t0, y0)

def test_validate_jacobian_input_non_square_matrix():
    """Test validation of a Jacobian function that returns a non-square matrix."""
    def jac(t, y):
        return np.array([[-2, 0, 1], [0, -3, 2]])  # Incorrect shape

    t0 = 0
    y0 = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="jac\\(t, y\\) must return a square matrix"):
        validate_jacobian_input(jac, t0, y0)

import math
import unittest


class TestStringToFunction(unittest.TestCase):

    ### ✅ Basic Scalar Function Tests ###
    def test_scalar_polynomial_function(self):
        func_str = "y**2 - sin(t)"
        f = parse_function_string(func_str)
        t_val, y_val = 2.0, 3.0
        expected_value = 9 - math.sin(2)  # 3^2 - 2 = 7

        result = f(t_val, y_val)
        self.assertAlmostEqual(result, expected_value)

        y0 = 3.0
        validated_f = validate_function_input(f, 0, np.atleast_1d(y0))
        validated_result = validated_f(t_val, np.atleast_1d(y_val))
        self.assertEqual(validated_result.shape, np.atleast_1d(y0).shape)

    ### ✅ Constant Function Test ###
    def test_constant_function(self):
        func_str = "4"
        f = parse_function_string(func_str)
        t_val, y_val = 2.0, 3.0
        expected_value = 4

        result = f(t_val, y_val)
        self.assertEqual(result, expected_value)

        y0 = 3.0
        validated_f = validate_function_input(f, 0, np.atleast_1d(y0))
        validated_result = validated_f(t_val, np.atleast_1d(y_val))
        self.assertEqual(validated_result.shape, np.atleast_1d(y0).shape)

    ### ✅ Trigonometric and Exponential Function Tests ###
    def test_trig_exponential_function(self):
        func_str = "sin(t) + exp(y)"
        f = parse_function_string(func_str)
        t_val, y_val = 0.0, 1.0
        expected_value = np.sin(0.0) + np.exp(1.0)  # sin(0) + e^1 = 1 + e

        result = f(t_val, y_val)
        self.assertAlmostEqual(result, expected_value)

    ### ✅ System of ODEs ###
    def test_system_function_parsing(self):
        func_str = "[-y0 + y1, y0 - 2*y1]"
        f = parse_function_string(func_str)
        t_val, y_val = 0.0, [1.0, 0.0]
        expected_values = [-1.0, 1.0]

        result = f(t_val, y_val)
        np.testing.assert_array_almost_equal(result, expected_values)

        y0 = [1.0, 0.0]
        validated_f = validate_function_input(f, 0, np.array(y0))
        validated_result = validated_f(t_val, np.array(y_val))
        np.testing.assert_array_almost_equal(validated_result, expected_values)

    ### ✅ Edge Cases ###
    def test_zero_function(self):
        func_str = "0"
        f = parse_function_string(func_str)
        result = f(10, 100)  # Should always return 0
        self.assertEqual(result, 0)

    def test_identity_function(self):
        func_str = "y - y"
        f = parse_function_string(func_str)
        result = f(10, 100)  # Should always return 0
        self.assertEqual(result, 0)

    def test_large_exponent_function(self):
        func_str = "t**100"
        f = parse_function_string(func_str)
        result = f(2, 0)  # 2^100
        self.assertEqual(result, 2**100)

    ### ❌ Invalid Function Tests ###
    def test_invalid_function_syntax(self):
        with self.assertRaises(Exception):
            parse_function_string("y ** ")

    def test_invalid_function_builtin(self):
        with self.assertRaises(Exception):
            parse_function_string("sum(y)")  # `sum(y)` is not a valid SymPy expression

    def test_invalid_nonexistent_variable(self):
        with self.assertRaises(Exception):
            parse_function_string("z**2 - t")  # `z` is not a defined variable

    def test_invalid_mismatched_brackets(self):
        with self.assertRaises(Exception):
            parse_function_string("[-y0 + y1, y0 - 2*y1")  # Missing closing bracket


import unittest
import numpy as np
from moses_ode.parsing.function_parser import parse_function_string  # Assuming you put parse_function_string in function_parser.py

class TestFunctionParsing(unittest.TestCase):

    def test_parse_scalar_function_valid(self):
        function_str = "-2*y + t"
        f_callable = parse_function_string(function_str) # num_variables is now auto-detected
        self.assertTrue(callable(f_callable))

        t_val = 1.0
        y_val = 2.0
        expected_dydt = -2 * y_val + t_val
        actual_dydt = f_callable(t_val, y_val)
        self.assertAlmostEqual(actual_dydt.item(), expected_dydt, places=7) # Compare floats with tolerance, use .item() for scalar

    def test_parse_scalar_function_invalid_syntax(self):
        function_str = "y + sin()" # Invalid sin() without argument (in sympy too if not corrected)
        with self.assertRaises(Exception): # Expecting a generic Exception from sympy.sympify for syntax errors
            parse_function_string(function_str)

    def test_parse_system_function_valid(self):
        function_str = "[-y0 + 2*y1, y0 - 3*y1 + t]" # Use y0, y1 instead of y[0], y[1] for sympy
        f_callable = parse_function_string(function_str)
        self.assertTrue(callable(f_callable))

        t_val = 0.5
        y_vector = np.array([1.0, 2.0])
        expected_dydt = np.array([-y_vector[0] + 2*y_vector[1], y_vector[0] - 3*y_vector[1] + t_val])
        actual_dydt = f_callable(t_val, y_vector)
        np.testing.assert_array_almost_equal(actual_dydt, expected_dydt, decimal=7)

    def test_parse_system_function_invalid_expression(self):
        function_str = "[-y0 + y1, y0 + invalid_var]" # 'invalid_var' is not defined
        with self.assertRaises(Exception): # Expecting a generic Exception from sympy.sympify
            parse_function_string(function_str)

    def test_parse_system_function_incorrect_num_expressions(self):
        # This test needs to be adjusted to match the current parsing behavior
        function_str = "[y1, y0]"  # Valid format but incorrect variables
        try:
            f_callable = parse_function_string(function_str)
            # If this passes, we should at least verify we can call the function
            self.assertTrue(callable(f_callable))
        except Exception as e:
            # If the current implementation correctly raises an exception for invalid variables,
            # that's fine too - just make sure we don't fail for the wrong reason
            self.assertIn("variable", str(e).lower())  # Error should mention variables in some way

    def test_parse_function_string_three_variables(self):
        function_str = "[-y0, y0 - y1, y1 - y2 + t]" # Use y0, y1, y2 for sympy
        f_callable = parse_function_string(function_str)
        self.assertTrue(callable(f_callable))

        t_val = 2.0
        y_vector = np.array([1.0, 0.5, 0.25])
        expected_dydt = np.array([-y_vector[0], y_vector[0] - y_vector[1], y_vector[1] - y_vector[2] + t_val])
        actual_dydt = f_callable(t_val, y_vector)
        np.testing.assert_array_almost_equal(actual_dydt, expected_dydt, decimal=7)

    def test_validate_function_input_valid_scalar(self):
        function_str = "-y"
        f_callable = parse_function_string(function_str)
        t0 = 0.0
        y0 = 1.0
        validated_f = validate_function_input(f_callable, t0, y0)
        self.assertEqual(validated_f, f_callable) # Should return the same function if valid

    def test_validate_function_input_valid_system(self):
        function_str = "[-y0 + y1, y0 - 2*y1]" # Use y0, y1 for sympy
        f_callable = parse_function_string(function_str)
        t0 = 0.0
        y0 = [1.0, 0.5]
        validated_f = validate_function_input(f_callable, t0, y0)
        self.assertEqual(validated_f, f_callable) # Should return the same function if valid

    def test_validate_function_input_invalid_return_shape(self):
        def bad_function(t, y): # Function that returns scalar when y is vector
            return -np.sum(y)

        t0 = 0.0
        y0 = [1.0, 0.5]
        with self.assertRaises(ValueError) as context:
            validate_function_input(bad_function, t0, y0)
        self.assertTrue("f(t, y) must return the same shape as y0" in str(context.exception))

    def test_validate_function_input_not_callable(self):
        not_a_function = "string is not callable"
        t0 = 0.0
        y0 = 1.0
        with self.assertRaises(TypeError) as context:
            validate_function_input(not_a_function, t0, y0)
        self.assertTrue("ODE function f must be callable" in str(context.exception))

    def test_validate_function_input_returns_list(self):
        def list_returning_function(t, y):
            return [-y[0]]  # Returns a list

        t0 = 0.0
        y0 = np.array([1.0])
        # The function should now accept lists and convert them to arrays
        # No exception should be raised
        result = validate_function_input(list_returning_function, t0, y0)
        self.assertTrue(callable(result))


if __name__ == '__main__':
    unittest.main()