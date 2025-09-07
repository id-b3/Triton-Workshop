#!/usr/bin/env python3
"""Solution for Exercise 4: Tiled Matrix Multiplication."""


# Comment out to run on GPU
# os.environ["TRITON_INTERPRETER"] = "1"

import torch
import triton
import triton.language as tl
from rich import print


# --- Autotune Example (Bonus) ---
# The @triton.autotune decorator can benchmark different configurations
# and automatically pick the fastest one for a given problem size.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 2}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """
    Tiled matrix multiplication kernel (C = A x B).
    
    Each program computes a BLOCK_SIZE_M x BLOCK_SIZE_N tile of C.
    The kernel iterates over the K dimension in blocks of size BLOCK_SIZE_K.
    """
    # 1. Get program IDs for the 2D grid
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Grouped ordering for better L2 cache performance
    pid_m = (pid // num_pid_n) % num_pid_m
    pid_n = pid % num_pid_n
    
    # 2. Create ranges for the M and N dimensions of the C tile
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    
    # 3. Initialize accumulator with zeros
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 4. Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Create range for the K dimension block
        offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        
        # Calculate pointers to the current blocks of A and B
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        
        # Create masks to prevent out-of-bounds access
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        # Load blocks of A and B from memory
        a_block = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Compute dot product and accumulate
        accumulator += tl.dot(a_block, b_block, allow_tf32=False)
        
    # 5. Calculate output pointers and mask
    offs_c_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_c_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_c_m[:, None] + stride_cn * offs_c_n[None, :]
    c_mask = (offs_c_m[:, None] < M) & (offs_c_n[None, :] < N)
    
    # 6. Store the result tile to C
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper function for the matrix multiplication kernel."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
    
    M, K = a.shape
    K, N = b.shape
    
    # Create output tensor
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Use a lambda function for dynamic grid calculation
    # This is useful for autotuning, as 'meta' contains block sizes
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def test_matmul():
    """Test suite for the matmul kernel."""
    test_cases = [
        # (M, N, K)
        (128, 256, 512),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (1000, 2000, 3000),
        (4096, 4096, 4096),
    ]
    
    for M, N, K in test_cases:
        print(f"\nTesting MatMul with M={M}, N={N}, K={K}")
        a = torch.randn((M, K), device='cuda', dtype=torch.float32)
        b = torch.randn((K, N), device='cuda', dtype=torch.float32)
        
        triton_result = matmul(a, b)
        torch_result = torch.matmul(a, b)
        best_config = matmul_kernel.best_config
        
        assert torch.allclose(triton_result, torch_result, atol=1e-2, rtol=1e-3), \
            f"Failed for size ({M}, {N}, {K})"
        print(f"✅ Test passed for size ({M}, {N}, {K})")
        print(f"   -> [cyan]Autotuner chose best config:[/cyan] {best_config}")

    print("\n🎊 All matmul tests passed!")

if __name__ == "__main__":
    test_matmul()
