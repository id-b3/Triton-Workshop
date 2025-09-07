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
    # 1. Get program ID
    pid = tl.program_id(axis=0)

    # 2. Calculate offsets for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 3. Create a mask to guard memory access
    mask = offsets < n_elements

    # 4. Load x and y vectors safely using the mask
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 5. Compute the element-wise sum
    output = x + y

    # 6. Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    # Manual grid calculation
    # BLOCK_SIZE = 1024
    # num_elements = x.numel()
    # num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
    #
    # Grid is just a tuple of integers
    # grid = (num_blocks,)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, x.numel(), BLOCK_SIZE=1024)
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
