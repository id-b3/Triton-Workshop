#!/usr/bin/env python3
"""Exercise 2: Sepia Image Filter."""

import os

# Set to "1" to run on CPU for debugging, comment out to run on GPU
# os.environ["TRITON_INTERPRETER"] = "1"

import torch
import triton
import triton.language as tl
from PIL import Image
from torchvision import transforms
from rich import print


@triton.jit
def sepia_kernel(
    rgb_ptr,          # Input RGB image pointer
    output_ptr,       # Output Sepia image pointer
    height,           # Image height
    width,            # Image width
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    """
    Triton kernel to apply a sepia filter to an RGB image.

    The output is also an RGB image, with each channel's value calculated
    based on a weighted sum of the original R, G, and B values.
    """
    # TODO 1: Get the 2D program ID for this block (row-block and column-block).
    pid_h = # YOUR CODE HERE
    pid_w = # YOUR CODE HERE

    # TODO 2: Calculate the 2D grid of linear memory offsets for the pixels
    # this program is responsible for.
    # HINT: First create 1D ranges for rows and columns, then combine them.
    row_offsets = # YOUR CODE HERE
    col_offsets = # YOUR CODE HERE
    pixel_offsets = # YOUR CODE HERE

    # TODO 3: Create a 2D mask to prevent out-of-bounds memory access.
    # HINT: Use the row and column offsets and compare them to height and width.
    mask = # YOUR CODE HERE

    # The image is stored as [R channel, G channel, B channel].
    # This is the size of a single color channel in memory.
    channel_size = height * width

    # TODO 4: Load the R, G, and B channels using the offsets and mask.
    # HINT: Use tl.load() with the `mask` and `other=0.0` arguments.
    r = # YOUR CODE HERE (load from rgb_ptr + 0 * channel_size)
    g = # YOUR CODE HERE (load from rgb_ptr + 1 * channel_size)
    b = # YOUR CODE HERE (load from rgb_ptr + 2 * channel_size)

    # TODO 5: Compute the sepia transformation using the formulas below.
    # HINT: Use tl.minimum() to clamp the results at 1.0.
    #
    # r_sepia = 0.393 * r + 0.769 * g + 0.189 * b
    # g_sepia = 0.349 * r + 0.686 * g + 0.168 * b
    # b_sepia = 0.272 * r + 0.534 * g + 0.131 * b
    r_sepia = # YOUR CODE HERE
    g_sepia = # YOUR CODE HERE
    b_sepia = # YOUR CODE HERE

    # TODO 6: Store the new R, G, B values back to the output tensor.
    # You will need three tl.store() calls, one for each channel.
    # YOUR CODE HERE (Store Red channel)
    # YOUR CODE HERE (Store Green channel)
    # YOUR CODE HERE (Store Blue channel)


def apply_sepia_filter(image_tensor: torch.Tensor) -> torch.Tensor:
    """Wrapper function to apply a sepia filter using the Triton kernel."""
    assert image_tensor.is_cuda and image_tensor.is_contiguous(), "Input must be a contiguous CUDA tensor"
    assert image_tensor.shape[0] == 3, "Input image must have 3 channels (RGB)"

    channels, height, width = image_tensor.shape
    output_tensor = torch.empty_like(image_tensor)
    BLOCK_H, BLOCK_W = 32, 32
    grid = (triton.cdiv(height, BLOCK_H), triton.cdiv(width, BLOCK_W))

    sepia_kernel[grid](
        image_tensor, output_tensor,
        height, width,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )
    return output_tensor


def test_sepia_filter():
    """Test suite and visualizer for the sepia filter kernel."""
    print("\n[bold]Testing Sepia Filter[/bold]")
    try:
        img = Image.open('slides/poseidon_triton.jpeg').convert('RGB')
        transform = transforms.ToTensor()
        image_tensor = transform(img).cuda().contiguous()
        print(f"Loaded 'poseidon_triton.jpeg' with shape: {image_tensor.shape}")
    except FileNotFoundError:
        print("[yellow]Warning: 'slides/poseidon_triton.jpeg' not found. Using a random tensor.[/yellow]")
        image_tensor = torch.rand(3, 256, 384, device='cuda', dtype=torch.float32).contiguous()

    # Apply the sepia filter using your Triton kernel
    sepia_image_tensor = apply_sepia_filter(image_tensor)

    print("✅ Kernel executed. If your implementation is correct, you should see a sepia-toned image.")
    print("Displaying original and sepia-filtered images for visual comparison.")

    # Convert tensors back to PIL Images for display
    original_pil = transforms.ToPILImage()(image_tensor.cpu())
    sepia_pil = transforms.ToPILImage()(sepia_image_tensor.cpu())

    # Show the images
    original_pil.show(title="Original Image")
    sepia_pil.show(title="Sepia Filtered Image (Your Implementation)")


if __name__ == "__main__":
    test_sepia_filter()
