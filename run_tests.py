#!/usr/bin/env python3
"""
Test runner script for pymaftools
"""

import sys
import subprocess
import os
from pathlib import Path

def run_tests(test_type="all", verbose=True, coverage=False):
    """
    Run tests for pymaftools
    
    Parameters:
    -----------
    test_type : str
        Type of tests to run: 'all', 'core', 'plot', 'model', 'fast', 'slow'
    verbose : bool
        Whether to run tests in verbose mode
    coverage : bool
        Whether to generate coverage report
    """
    
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Change to project directory
    os.chdir(project_root)
    
    # Build pytest command
    cmd = ["/home/data/data_dingyangliu/miniconda3/envs/bioinfo/bin/python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=pymaftools", "--cov-report=html", "--cov-report=term"])
    
    # Add test selection based on type
    if test_type == "core":
        cmd.append("tests/core/")
    elif test_type == "plot":
        cmd.append("tests/plot/")
    elif test_type == "model":
        cmd.append("tests/model/")
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    elif test_type == "slow":
        cmd.extend(["-m", "slow"])
    elif test_type == "all":
        cmd.append("tests/")
    else:
        print(f"Unknown test type: {test_type}")
        return 1
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print("pytest not found. Please install pytest:")
        print("pip install pytest pytest-cov")
        return 1


def main():
    """Main entry point for test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run pymaftools tests")
    parser.add_argument(
        "--type", 
        choices=["all", "core", "plot", "model", "fast", "slow"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Run tests in quiet mode"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true", 
        help="Generate coverage report"
    )
    
    args = parser.parse_args()
    
    return run_tests(
        test_type=args.type,
        verbose=not args.quiet,
        coverage=args.coverage
    )


if __name__ == "__main__":
    sys.exit(main())