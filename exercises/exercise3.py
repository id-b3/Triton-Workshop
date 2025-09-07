#!/usr/bin/env python3
"""Exercise 3: 2D Convolution."""

import os

# Set to "1" to run on CPU for debugging, comment out to run on GPU
# os.environ["TRITON_INTERPRETER"] = "1"

import torch
import triton
import triton.language as tl
from PIL import Image
from torchvision import transforms
from rich import print

# --- Helper Function ---
@triton.jit
def _get_2d_offsets_and_mask(pid_h, pid_w, height, width, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    """
    Helper JIT function to calculate 2D offsets and a boundary mask.
    This keeps the main kernel cleaner.
    """
    # TODO 1: Calculate the 1D offset ranges for this block's rows and columns.
    # HINT: This is very similar to the 2D exercises.
    # Create a range from 0 to BLOCK_H/BLOCK_W and shift it by the block's start position.
    offsets_h = # YOUR CODE HERE
    offsets_w = # YOUR CODE HERE
    
    # TODO 2: Create the 2D boundary mask.
    # HINT: Use broadcasting `[:, None]` and `[None, :]` on the offsets.
    mask = # YOUR CODE HERE
    
    return offsets_h, offsets_w, mask

# --- Main Kernel ---
@triton.jit
def conv2d_kernel(
    input_ptr, output_ptr, kernel_ptr,
    height, width,
    kernel_size: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    """
    Triton kernel for a 2D convolution operation.
    This kernel uses a 3D grid: (channels, height_blocks, width_blocks).
    """
    # TODO 3: Get the program ID for each of the 3 dimensions (channel, height, width).
    # Remember axis=0, axis=1, axis=2.
    pid_c = # YOUR CODE HERE
    pid_h = # YOUR CODE HERE
    pid_w = # YOUR CODE HERE
    
    # We call the helper function you will complete to get our block's offsets and mask.
    offsets_h, offsets_w, mask = _get_2d_offsets_and_mask(
        pid_h, pid_w, height, width, BLOCK_H, BLOCK_W
    )

    # TODO 4: Initialize an accumulator for this block with zeros.
    # The shape should be (BLOCK_H, BLOCK_W).
    accumulator = # YOUR CODE HERE
    
    # Calculate the offset for the current channel.
    channel_offset = pid_c * height * width
    
    # Iterate over the convolution kernel (e.g., a 3x3 window)
    for kh in tl.static_range(kernel_size):
        for kw in tl.static_range(kernel_size):
            # TODO 5: Calculate the offsets into the *input* image.
            # This is the "sliding window": current output pixel's position + kernel's offset.
            input_h = # YOUR CODE HERE
            input_w = # YOUR CODE HERE
            
            # TODO 6: Create a new `valid_mask` to ensure the input offsets are within bounds.
            # This implements "padding" by ensuring we don't read outside the image.
            # Combine the original `mask` with checks for `input_h` and `input_w`.
            valid_mask = # YOUR CODE HERE
            
            # Calculate the linear memory index for the input pixels
            input_indices = channel_offset + input_h * width + input_w
            
            # TODO 7: Load the current kernel weight and the corresponding input pixel value.
            # Use the `valid_mask` and `other=0.0` when loading from the input tensor.
            kernel_val = # YOUR CODE HERE
            input_val = # YOUR CODE HERE
            
            # TODO 8: Perform the multiply-accumulate operation.
            # YOUR CODE HERE

    # TODO 9: Store the final result from the accumulator back to the output tensor.
    # Calculate the output indices and use the original `mask` from the helper function.
    output_indices = channel_offset + offsets_h[:, None] * width + offsets_w[None, :]
    # YOUR CODE HERE


def apply_convolution(image_tensor: torch.Tensor, kernel: list) -> torch.Tensor:
    """Wrapper function to apply a convolution using the Triton kernel."""
    assert image_tensor.is_cuda and image_tensor.is_contiguous(), "Input must be a contiguous CUDA tensor"
    
    channels, height, width = image_tensor.shape
    output_tensor = torch.empty_like(image_tensor)
    
    # Convert kernel to a contiguous CUDA tensor
    kernel_tensor = torch.tensor(kernel, dtype=torch.float32, device='cuda').contiguous()
    kernel_size = kernel_tensor.shape[0]
    
    # Define block sizes for processing
    BLOCK_H, BLOCK_W = 16, 16
    
    # Define the 3D grid
    grid = (channels, triton.cdiv(height, BLOCK_H), triton.cdiv(width, BLOCK_W))
    
    conv2d_kernel[grid](
        image_tensor, output_tensor, kernel_tensor,
        height, width,
        kernel_size=kernel_size,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )
    
    return output_tensor


def test_convolution():
    """Test suite for the convolution kernel."""
    print("\nTesting 2D Convolution")
    try:
        img = Image.open('./slides/poseidon_triton.jpeg').convert('RGB')
        transform = transforms.ToTensor()
        image_tensor = transform(img).cuda().contiguous()
    except FileNotFoundError:
        print("Warning: 'slides/poseidon_triton.jpeg' not found. Using a random tensor.")
        image_tensor = torch.randn(3, 256, 256, device='cuda', dtype=torch.float32).contiguous()

    # Define Sobel kernel for edge detection
    x_sobel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    
    print("Applying X Sobel filter...")
    triton_result = apply_convolution(image_tensor, x_sobel)
        
    transforms.ToPILImage()(triton_result.cpu().clamp(0, 1)).show()

if __name__ == "__main__":
    test_convolution()
