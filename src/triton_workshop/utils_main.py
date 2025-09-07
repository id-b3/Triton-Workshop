# Shared utilities for the Triton Attention Workshop
# Common functions used across all sessions

import triton
import sys

from rich import print


def setup_check():
    """Comprehensive environment verification for workshop participants."""
    print("=== TRITON WORKSHOP ENVIRONMENT CHECK ===\n")

    success = True

    # Check Python version
    if sys.version_info >= (3, 12):
        print("✅ Python 3.12+ detected")
    else:
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} detected, need 3.12+")
        success = False

    # Test imports
    imports_to_test = [("torch", "PyTorch"), ("triton", "Triton"), ("numpy", "NumPy"), ("matplotlib", "Matplotlib")]

    for module, name in imports_to_test:
        try:
            __import__(module)
            print(f"✅ {name} available")
        except ImportError:
            print(f"❌ {name} not found")
            success = False

    # Test CUDA if available
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ CUDA GPU: {gpu_name} ({memory_gb:.1f}GB)")

            # Test Triton compilation
            @triton.jit
            def test_kernel(x_ptr, n_elements, BLOCK_SIZE: triton.language.constexpr):
                pid = triton.language.program_id(0)
                offsets = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = triton.language.load(x_ptr + offsets, mask=mask)
                triton.language.store(x_ptr + offsets, x, mask=mask)

            # Test compilation
            test_input = torch.randn(1024, device="cuda")
            grid = (triton.cdiv(1024, 256),)
            test_kernel[grid](test_input, 1024, BLOCK_SIZE=256)
            print("✅ Triton compiler working")

        else:
            print("⚠️  No CUDA GPU detected (interpreter mode available)")
            print("💡 Set TRITON_INTERPRET_MODE=1 for CPU debugging")
    except Exception as e:
        print(f"❌ CUDA/Triton test failed: {e}")
        success = False

    if success:
        print("\n🎉 Environment check completed successfully!")
        print("   Ready to start the workshop!")
        print("\n🚀 Next steps:")
        print("   launch the presentation with `presenterm slides/workshop_slides.md`")
        print("   uv run exercise1  # Start with Exercise 1")
    else:
        print("\n❌ Environment setup incomplete")
        print("   Please fix the issues above before starting")

    return success


if __name__ == "__main__":
    setup_check()
