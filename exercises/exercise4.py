#!/usr/bin/env python3
"""Exercise 4: Tiled Matrix Multiplication."""

import os

# Comment out to run on GPU
# os.environ["TRITON_INTERPRETER"] = "1"

import torch
import triton
import triton.language as tl
from rich import print


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    """
    Tiled matrix multiplication kernel (C = A x B).
    
    Each program computes a BLOCK_SIZE_M x BLOCK_SIZE_N tile of C.
    The kernel iterates over the K dimension in blocks of size BLOCK_SIZE_K.
    """
    # TODO 1: Get the 2D program IDs for the M (row) and N (column) dimensions.
    pid_m = # YOUR CODE HERE
    pid_n = # YOUR CODE HERE
    
    # TODO 2: Create the ranges for the M and N dimensions of the C tile this program handles.
    # This will be a 1D vector of offsets for rows and columns.
    offs_m = # YOUR CODE HERE
    offs_n = # YOUR CODE HERE

    # TODO 3: Initialize an accumulator tile with zeros.
    # Its shape should be (BLOCK_SIZE_M, BLOCK_SIZE_N).
    accumulator = # YOUR CODE HERE

    # Loop over the K dimension in blocks of size BLOCK_SIZE_K
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # TODO 4: Create the range for the current block of the K dimension.
        offs_k = # YOUR CODE HERE
        
        # TODO 5: Calculate the pointers to the current blocks of A and B.
        # Use broadcasting with the M, N, K offsets and the strides.
        # Shape of a_ptrs should be (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # Shape of b_ptrs should be (BLOCK_SIZE_K, BLOCK_SIZE_N)
        a_ptrs = # YOUR CODE HERE
        b_ptrs = # YOUR CODE HERE
        
        # TODO 6: Create masks to prevent out-of-bounds access for A and B.
        a_mask = # YOUR CODE HERE
        b_mask = # YOUR CODE HERE
        
        # TODO 7: Load the blocks of A and B using the pointers and masks.
        # Use other=0.0 to handle boundary conditions.
        a_block = # YOUR CODE HERE
        b_block = # YOUR CODE HERE
        
        # TODO 8: Compute the matrix multiplication for the blocks and accumulate the result.
        # Use tl.dot().
        # YOUR CODE HERE
        
    # TODO 9: After the loop, calculate the pointers to the C output tile.
    c_ptrs = # YOUR CODE HERE
    c_mask = # YOUR CODE HERE
    
    # TODO 10: Store the final accumulated result to the C matrix.
    # YOUR CODE HERE


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper function for the matrix multiplication kernel."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
    
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Grid is 2D, corresponding to the C matrix tiles
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32
    )
    return c


def test_matmul():
    """Test suite for the matmul kernel."""
    M, N, K = 512, 512, 512
    print(f"\nTesting MatMul with M={M}, N={N}, K={K}")
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    
    triton_result = matmul(a, b)
    torch_result = torch.matmul(a, b)
    
    assert torch.allclose(triton_result, torch_result, atol=1e-2, rtol=1e-3), "Matmul result mismatch!"
    print(f"✅ Matmul test passed for size ({M}, {N}, {K})")

if __name__ == "__main__":
    test_matmul()
