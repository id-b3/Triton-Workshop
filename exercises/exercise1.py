#!/usr/bin/env python3
"""First exercise from the Triton workshop."""

import os

# HINT: set TRITON_INTERPRETER before importing triton for debugging
# Use some ready-made debugging functions from triton_workshop.utils_debug
os.environ["TRITON_INTERPRETER"] = "1"
import torch
import triton
import triton.language as tl

from rich import print


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for element-wise vector addition.

    Each program (block) processes BLOCK_SIZE elements.
    Multiple programs run in parallel to handle the full tensor.
    """

    # TODO 1: Get the program ID for this block
    pid = # YOUR CODE HERE

    # TODO 2: Calculate memory offsets for this block
    offsets = # YOUR CODE HERE

    # TODO 3: Create a mask to prevent out-of-bounds memory access
    # HINT: Some offsets might exceed n_elements, especially in the last block
    mask = # YOUR CODE HERE

    # TODO 4: Load data from memory using the calculated offsets
    # HINT: Use tl.load() with the mask parameter for safety
    x = # YOUR CODE HERE
    y = # YOUR CODE HERE

    # TODO 5: Perform the addition operation
    output = # YOUR CODE HERE

    # TODO 6: Store the result back to memory
    # YOUR CODE HERE


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton kernel.

    Args:
        x, y: Input tensors to add (must be same shape)

    Returns:
        Sum of x and y
    """

    # TODO 7: Create output tensor with same shape as input
    output = # YOUR CODE HERE

    # TODO 8: Define the grid (number of blocks to launch)
    # HINT: Use triton.cdiv() to calculate how many blocks are needed or calculate manually
    # HINT: Grid is either a function that takes meta (kernel metadata) and returns tuple, or a tuple
    grid = lambda meta: (# YOUR CODE HERE,)

    # TODO 9: Launch the kernel with appropriate arguments
    # HINT: Syntax is kernel[grid](arg1, arg2, ..., BLOCK_SIZE=1024)
    # YOUR CODE HERE

    return output


def test_triton_addition():
    """Test suite for the addition kernel."""
    test_cases = [
        (100, "Small tensor"),
        (1024, "Single block"),
        (2048, "Two blocks"),  
        (5000000, "Large tensor"),
        (5000001, "Odd size (tests masking)"),
    ]

    for size, description in test_cases:
        print(f"\nTesting {description} (size: {size})")

        x = torch.randn(size).cuda()
        y = torch.randn(size).cuda()

        # Run Triton kernel
        triton_result = add(x, y)
        # Compare with PyTorch
        pytorch_result = x + y

        assert torch.allclose(triton_result, pytorch_result, rtol=1e-5, atol=1e-8), \
            f"Failed for {description}"
        print(f"✅ {description} passed")

    print("\n🎉 All tests passed!")

test_triton_addition()
