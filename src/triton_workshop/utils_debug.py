"""
Triton Workshop Debug Utilities

A utility module for Triton GPU kernel development workshops, providing
debugging helpers, tensor validation, and common utility functions.
"""

import os
from typing import Any, List

# This must be set before importing triton for proper debugging support
os.environ["TRITON_INTERPRET"] = "1"


def check_tensors_gpu_ready(*tensors: Any) -> None:
    """
    Validate that tensors are ready for GPU computation.

    Checks that all provided tensors are:
    - Contiguous in memory
    - Located on CUDA device (unless in interpretation mode)

    Args:
        *tensors: Variable number of tensor objects to validate

    Raises:
        AssertionError: If any tensor fails validation checks
    """
    for tensor in tensors:
        assert tensor.is_contiguous(), "Tensor is not contiguous in memory"

        # Skip CUDA check when running in interpretation mode
        if os.environ.get("TRITON_INTERPRET") != "1":
            assert tensor.is_cuda, "Tensor is not on CUDA device"


def test_pid_conditions(conditions: str, pid_0: List[int], pid_1: List[int] | None, pid_2: List[int] | None) -> bool:
    """
    Test if process ID conditions are fulfilled for kernel debugging.

    Evaluates conditions on program instance IDs (PIDs) used in Triton kernels.
    Conditions are specified as comma-separated comparisons.

    Args:
        conditions: Comma-separated condition string (e.g., "=0", ">1,=0")
        pid_0: List containing pid_0 value [default: [0]]
        pid_1: List containing pid_1 value [default: [0]]
        pid_2: List containing pid_2 value [default: [0]]

    Returns:
        bool: True if all specified conditions are met, False otherwise

    Examples:
        >>> test_pid_conditions("=0")  # pid_0 == 0
        True
        >>> test_pid_conditions(">1,=0", [2], [0])  # pid_0 > 1 and pid_1 == 0
        True
        >>> test_pid_conditions("<=5,!=3", [4], [2])  # pid_0 <= 5 and pid_1 != 3
        True

    Raises:
        ValueError: If condition uses invalid operator
    """
    # Default values for PIDs
    if pid_1 is None:
        pid_1 = [0]
    if pid_2 is None:
        pid_2 = [0]

    pids = (pid_0[0], pid_1[0], pid_2[0])
    condition_list = conditions.replace(" ", "").split(",")
    valid_operators = {"<", ">", ">=", "<=", "=", "!="}

    for i, condition in enumerate(condition_list):
        if not condition:  # Skip empty conditions
            continue

        if i >= len(pids):  # Skip if no corresponding PID
            continue

        # Parse operator and threshold
        operator = ""
        threshold_str = ""

        # Handle multi-character operators first
        if condition.startswith(">=") or condition.startswith("<=") or condition.startswith("!="):
            operator = condition[:2]
            threshold_str = condition[2:]
        elif condition[0] in valid_operators:
            operator = condition[0]
            threshold_str = condition[1:]
        else:
            raise ValueError(f"Invalid condition format: '{condition}'. Must start with one of: {valid_operators}")

        if not threshold_str.lstrip("-").isdigit():
            raise ValueError(f"Invalid threshold in condition: '{condition}'")

        try:
            threshold = int(threshold_str)
        except ValueError:
            raise ValueError(f"Cannot parse threshold in condition: '{condition}'")

        if operator not in valid_operators:
            raise ValueError(f"Invalid operator '{operator}'. Valid operators: {valid_operators}")

        # Convert '=' to '==' for evaluation
        if operator == "=":
            operator = "=="

        # Safely evaluate condition using comparison
        pid_value = pids[i]
        condition_met = _evaluate_condition(pid_value, operator, threshold)

        if not condition_met:
            return False

    return True


def _evaluate_condition(pid_value: int, operator: str, threshold: int) -> bool:
    """Safely evaluate a single PID condition without using eval."""
    if operator == "==":
        return pid_value == threshold
    elif operator == "!=":
        return pid_value != threshold
    elif operator == "<":
        return pid_value < threshold
    elif operator == ">":
        return pid_value > threshold
    elif operator == "<=":
        return pid_value <= threshold
    elif operator == ">=":
        return pid_value >= threshold
    else:
        return False


def print_if(
    text: str, pid_0: List[int], pid_1: List[int] | None = None, pid_2: List[int] | None = None, conditions: str = "<3"
) -> None:
    """
    Print text if PID conditions are met.

    Useful for conditional logging in kernel development.

    Args:
        text: Text to print
        conditions: Condition string to evaluate
        pid_0: List containing pid_0 value [default: [0]]
        pid_1: List containing pid_1 value [default: [0]]
        pid_2: List containing pid_2 value [default: [0]]

    Example:
        >>> print_if("Debug info", "=0", [0])  # Print when pid_0 == 0
    """
    if test_pid_conditions(conditions, pid_0, pid_1, pid_2):
        print(text)


def ceiling_divide(dividend: int, divisor: int) -> int:
    """
    Perform ceiling division (round up division).

    Efficiently computes ⌈dividend / divisor⌉ using integer arithmetic.

    Args:
        dividend: The number to be divided
        divisor: The number to divide by

    Returns:
        int: The ceiling of dividend / divisor

    Examples:
        >>> ceiling_divide(10, 2)
        5
        >>> ceiling_divide(10, 3)
        4
        >>> ceiling_divide(7, 3)
        3

    Raises:
        ZeroDivisionError: If divisor is zero
    """
    if divisor == 0:
        raise ZeroDivisionError("Division by zero")

    return (dividend + divisor - 1) // divisor
