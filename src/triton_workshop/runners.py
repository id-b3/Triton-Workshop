"""
This module provides entry points for running workshop scripts via `uv run`.

It uses the `runpy` module to execute the exercise and solution scripts
as if they were run directly from the command line. This allows them to
behave as standalone scripts while being launchable through standard
Python packaging entry points defined in `pyproject.toml`.
"""
import runpy
import sys

def _run_script(script_path: str):
    """Helper function to execute a script by its path."""
    try:
        runpy.run_path(script_path, run_name="__main__")
    except FileNotFoundError:
        print(f"Error: Script not found at '{script_path}'", file=sys.stderr)
        print("Please ensure you are running this command from the root of the workshop repository.", file=sys.stderr)
        sys.exit(1)

# --- Exercise Runners ---

def run_exercise1():
    _run_script("exercises/exercise1.py")

def run_exercise2():
    _run_script("exercises/exercise2.py")

def run_exercise3():
    _run_script("exercises/exercise3.py")

def run_exercise4():
    _run_script("exercises/exercise4.py")

# --- Solution Runners ---

def run_solution1():
    _run_script("solutions/exercise1.py")

def run_solution2():
    _run_script("solutions/exercise2.py")

def run_solution3():
    _run_script("solutions/exercise3.py")

def run_solution4():
    _run_script("solutions/exercise4.py")
