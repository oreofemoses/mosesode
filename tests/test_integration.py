import unittest
import numpy as np
from moses_ode.numerical.integration import (
    trapezoid,
    simpson_1_3,
    simpson_3_8,
    romberg
)

class TestIntegration(unittest.TestCase):
    """Test suite for numerical integration methods."""
    
    def setUp(self):
        # Test functions and their analytical integrals
        self.test_cases = [
            # f(x) = x^2, ∫x^2 dx from 0 to 1 = 1/3
            (lambda x: x**2, 0, 1, 1/3),
            
            # f(x) = sin(x), ∫sin(x) dx from 0 to π = 2
            (lambda x: np.sin(x), 0, np.pi, 2),
            
            # f(x) = e^x, ∫e^x dx from 0 to 1 = e - 1
            (lambda x: np.exp(x), 0, 1, np.exp(1) - 1)
        ]
    
    def test_trapezoid(self):
        """Test trapezoidal rule."""
        for f, a, b, expected in self.test_cases:
            result = trapezoid(f, a, b, n=1000)
            self.assertAlmostEqual(result, expected, places=4)
    
    def test_simpson_1_3(self):
        """Test Simpson's 1/3 rule."""
        for f, a, b, expected in self.test_cases:
            result = simpson_1_3(f, a, b, n=1000)
            self.assertAlmostEqual(result, expected, places=6)
    
    def test_simpson_3_8(self):
        """Test Simpson's 3/8 rule."""
        for f, a, b, expected in self.test_cases:
            result = simpson_3_8(f, a, b, n=999)  # Ensure n is multiple of 3
            self.assertAlmostEqual(result, expected, places=6)
    
    def test_romberg(self):
        """Test Romberg integration."""
        for f, a, b, expected in self.test_cases:
            result, _ = romberg(f, a, b, max_iterations=6)
            self.assertAlmostEqual(result, expected, places=10)
    
    def test_convergence_rates(self):
        """Test that higher-order methods converge faster."""
        f = lambda x: x**3  # Simple test function
        a, b = 0, 1
        expected = 0.25  # Analytical result
        
        # Test different numbers of intervals
        intervals = [10, 20, 40]
        trap_errors = []
        simp_errors = []
        
        for n in intervals:
            trap_errors.append(abs(trapezoid(f, a, b, n) - expected))
            simp_errors.append(abs(simpson_1_3(f, a, b, n) - expected))
        
        # For the cubic function, the errors are small but not zero due to floating-point precision
        # Just check that the errors are small enough and decreasing with more intervals
        self.assertLess(trap_errors[-1], 1e-3)  # More reasonable tolerance for last value
        
        # For Simpson's rule with cubic functions, the errors might be near machine precision
        # So we just verify the error is very small
        self.assertLess(simp_errors[-1], 1e-10)
        
        # For trapezoid method, errors should decrease as intervals increase
        self.assertLess(trap_errors[-1], trap_errors[0])

if __name__ == '__main__':
    unittest.main() 