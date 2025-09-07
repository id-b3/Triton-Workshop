#!/usr/bin/env python3
"""Solution for Exercise 3: 2D Convolution."""

import torch
import triton
import triton.language as tl
from PIL import Image
from torchvision import transforms
from rich import print

# --- Helper Function Example ---
@triton.jit
def _get_2d_offsets_and_mask(pid_h, pid_w, height, width, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    """
    Helper JIT function to calculate 2D offsets and a boundary mask.
    This keeps the main kernel cleaner.
    """
    # Calculate the starting row and column for this block
    start_h = pid_h * BLOCK_H
    start_w = pid_w * BLOCK_W
    
    # Create 1D ranges for the height and width of the block
    range_h = tl.arange(0, BLOCK_H)
    range_w = tl.arange(0, BLOCK_W)
    
    # Calculate the full 2D offsets for every pixel in this block
    offsets_h = start_h + range_h
    offsets_w = start_w + range_w
    
    # Create the boundary mask (using broadcasting to make a 2D array)
    mask = (offsets_h[:, None] < height) & (offsets_w[None, :] < width)
    
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
    # 1. Get program IDs for all 3 dimensions
    pid_c = tl.program_id(0)  # Channel ID
    pid_h = tl.program_id(1)  # Height-block ID
    pid_w = tl.program_id(2)  # Width-block ID
    
    # 2. Use the helper function to get offsets and mask for the output block
    offsets_h, offsets_w, mask = _get_2d_offsets_and_mask(
        pid_h, pid_w, height, width, BLOCK_H, BLOCK_W
    )

    # 3. Initialize an accumulator for this block with zeros
    accumulator = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Calculate the offset for the current channel
    channel_offset = pid_c * height * width
    
    # 4. Iterate over the convolution kernel (e.g., a 3x3 or 5x5 window)
    # tl.static_range is used for loops where the bounds are known at compile time.
    for kh in tl.static_range(kernel_size):
        for kw in tl.static_range(kernel_size):
            # 5. Calculate input offsets for the current kernel position
            # This is the "sliding window" part
            input_h = offsets_h[:, None] + kh
            input_w = offsets_w[None, :] + kw
            
            # 6. Create a mask to handle image boundaries (padding)
            # Ensure we only load pixels that are within the image dimensions
            valid_mask = mask & (input_h >= 0) & (input_h < height) & \
                                (input_w >= 0) & (input_w < width)
            
            # Calculate the linear memory index for the input pixels
            input_indices = channel_offset + input_h * width + input_w
            
            # 7. Load kernel weight and input pixel values
            kernel_val = tl.load(kernel_ptr + kh * kernel_size + kw)
            # The 'other=0.0' argument implements zero-padding for free
            input_val = tl.load(input_ptr + input_indices, mask=valid_mask, other=0.0)
            
            # 8. Accumulate the product
            accumulator += input_val * kernel_val

    # 9. Store the final convolved result for this block
    output_indices = channel_offset + offsets_h[:, None] * width + offsets_w[None, :]
    tl.store(output_ptr + output_indices, accumulator, mask=mask)


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
        img = Image.open('slides/poseidon_triton.jpeg').convert('RGB')
        transform = transforms.ToTensor()
        image_tensor = transform(img).cuda().contiguous()
    except FileNotFoundError:
        print("Warning: 'slides/poseidon_triton.jpeg' not found. Using a random tensor.")
        image_tensor = torch.randn(3, 256, 256, device='cuda', dtype=torch.float32).contiguous()

    # Define Sobel kernels for edge detection
    x_sobel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    y_sobel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    
    print("Applying X Sobel filter...")
    triton_result_x = apply_convolution(image_tensor, x_sobel)

    print("Applying Y Sobel filter...")
    triton_result_y = apply_convolution(image_tensor, y_sobel)
        
    # Display the results
    combined_sobel = torch.sqrt(triton_result_x**2 + triton_result_y**2)
    transforms.ToPILImage()(combined_sobel.cpu().clamp(0, 1)).show()

if __name__ == "__main__":
    test_convolution()
