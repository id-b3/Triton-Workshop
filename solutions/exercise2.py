#!/usr/bin/env python3
"""Solution for Exercise 2: Sepia Image Filter."""


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
    # 1. Get the 2D program ID for this block
    pid_h = tl.program_id(0)  # Row-block ID
    pid_w = tl.program_id(1)  # Column-block ID

    # 2. Calculate memory offsets for this block
    # This creates a 2D grid of linear memory offsets for the pixels
    # this program is responsible for.
    row_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    col_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    pixel_offsets = row_offsets[:, None] * width + col_offsets[None, :]

    # 3. Create a mask to prevent out-of-bounds memory access
    mask = (row_offsets[:, None] < height) & (col_offsets[None, :] < width)

    # 4. Load R, G, B channels safely using the mask
    # The image is stored as [R channel, G channel, B channel]
    channel_size = height * width
    r = tl.load(rgb_ptr + 0 * channel_size + pixel_offsets, mask=mask, other=0.0)
    g = tl.load(rgb_ptr + 1 * channel_size + pixel_offsets, mask=mask, other=0.0)
    b = tl.load(rgb_ptr + 2 * channel_size + pixel_offsets, mask=mask, other=0.0)

    # 5. Compute the sepia transformation for each channel
    # The formulas involve a weighted sum of the original R, G, B values.
    # tl.minimum is used to clamp the result at 1.0, preventing color overflow.
    r_sepia = tl.minimum(0.393 * r + 0.769 * g + 0.189 * b, 1.0)
    g_sepia = tl.minimum(0.349 * r + 0.686 * g + 0.168 * b, 1.0)
    b_sepia = tl.minimum(0.272 * r + 0.534 * g + 0.131 * b, 1.0)

    # 6. Store the new R, G, B values back to the output tensor
    # The output has the same 3-channel layout as the input.
    tl.store(output_ptr + 0 * channel_size + pixel_offsets, r_sepia, mask=mask)
    tl.store(output_ptr + 1 * channel_size + pixel_offsets, g_sepia, mask=mask)
    tl.store(output_ptr + 2 * channel_size + pixel_offsets, b_sepia, mask=mask)


def apply_sepia_filter(image_tensor: torch.Tensor) -> torch.Tensor:
    """Wrapper function to apply a sepia filter using the Triton kernel."""
    assert image_tensor.is_cuda and image_tensor.is_contiguous(), "Input must be a contiguous CUDA tensor"
    assert image_tensor.shape[0] == 3, "Input image must have 3 channels (RGB)"

    channels, height, width = image_tensor.shape

    # Create an output tensor with the same shape, device, and dtype as the input
    output_tensor = torch.empty_like(image_tensor)

    # Define block sizes for processing
    BLOCK_H, BLOCK_W = 32, 32

    # Define the 2D grid of programs to launch
    grid = (triton.cdiv(height, BLOCK_H), triton.cdiv(width, BLOCK_W))

    # Launch the kernel
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

    # Apply the sepia filter using our Triton kernel
    sepia_image_tensor = apply_sepia_filter(image_tensor)

    print("✅ Kernel executed successfully.")
    print("Displaying original and sepia-filtered images for visual comparison.")

    # Convert tensors back to PIL Images for display
    original_pil = transforms.ToPILImage()(image_tensor.cpu())
    sepia_pil = transforms.ToPILImage()(sepia_image_tensor.cpu())

    # Show the images
    original_pil.show(title="Original Image")
    sepia_pil.show(title="Sepia Filtered Image (Triton)")


if __name__ == "__main__":
    test_sepia_filter()
