#!/usr/bin/env python3
import os
import sys
import subprocess

def run_tests():
    """Run all tests with coverage reporting"""
    print("Running tests with coverage reporting...")
    
    # Make sure we're in the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run pytest with coverage
    cmd = [
        "pytest",
        "tests/",
        "--cov=moses_ode",
        "--cov-report=term",
        "--cov-report=html:coverage_html",
        "-v"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    print(result.stdout)
    if result.stderr:
        print("ERRORS:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    
    # Print summary
    if result.returncode == 0:
        print("\nAll tests passed successfully!")
        print("\nCoverage report generated in coverage_html/ directory")
        print("Open coverage_html/index.html in a browser to view the detailed report")
    else:
        print(f"\nTests failed with return code {result.returncode}")
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests()) 