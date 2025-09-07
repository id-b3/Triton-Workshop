# A Hands-On Introduction to Triton

## 🚀 From Python to GPU Kernels in 3 Hours

This repository contains the code, exercises, and presentation for a hands-on workshop designed to teach you how to write custom, high-performance GPU kernels from scratch using OpenAI's Triton language.

### 🎯 What You'll Build

By the end of this workshop, you will have written and optimized several fundamental GPU kernels:

-   **Vector Addition:** Your first 1D kernel, mastering offsets and memory masking.
-   **Image Filters (Sepia & Greyscale):** Handle 2D data layouts, broadcasting, and multi-channel operations.
-   **2D Convolution:** Implement the core operation of CNNs with a 3D kernel grid and nested loops.
-   **Tiled Matrix Multiplication:** Build the workhorse of modern AI and learn performance-oriented tiling patterns.

## 🏗️ Workshop Structure

| Session                                                  | Duration  | Key Concepts Covered                                        |
| :------------------------------------------------------- | :-------- | :---------------------------------------------------------- |
| **1. The 'Why' & 'How' of Custom Kernels**               | (20 min)  | Motivation, Kernel Fusion, Triton vs. CUDA Programming Models |
| **2. Your First Kernel: 1D Vector Operations**           | (40 min)  | Program IDs, Offsets, Masking, Load/Store Operations        |
| **3. Thinking in 2D: Image Filtering**                   | (45 min)  | 2D Grids, Broadcasting, Multi-channel Data Layouts          |
| **4. Building Blocks of AI: Convolution & MatMul**       | (60 min)  | 3D Grids, Tiled Algorithms, Accumulators, `tl.dot`          |
| **5. Making it Fast: Performance & Next Steps**          | (15 min)  | Debugging with `TRITON_INTERPRETER`, `autotune`, Real-world usage |

## 🚀 Quick Start

### Prerequisites

-   **Python 3.12+**
-   An **NVIDIA GPU** with CUDA installed (recommended for performance).
-   [presenterm](https://github.com/mfontanini/presenterm) for slides and a terminal with [image/sixel](https://saitoha.github.io/libsixel/) support (e.g. Windows Terminal, Kitty, iTerm2)
-   CPU-only is possible for debugging using Triton's interpreter mode.

### Installation

The project uses `uv` for fast and reliable dependency management.
Alternatively install using `pip install -e .`.

```bash
# 1. Clone the repository
git clone https://github.com/id-b3/triton-workshop.git
cd triton-workshop

# 2. Install dependencies with uv (recommended)
# If you don't have uv: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 3. Verify your installation
uv run setup-check
```

A successful setup will produce the following output:
```
✅ Python 3.10+ detected
✅ PyTorch with CUDA support found
✅ Triton compiler is working
✅ All workshop dependencies installed
🚀 You are ready for the Triton workshop!
```

## 💻 Running the Exercises

This is a hands-on workshop. You will find templates in the `exercises/` directory and complete solutions in the `solutions/` directory.

Run each exercise script using `uv run`.

```bash
# Start with the first exercise
uv run exercise_1

# When you're ready, move to the next one
uv run exercise_2
...

# You can run the solution files to compare
uv run solution_1
```

## 🧪 Example Usage

Once you've completed Exercise 4, you'll have a fully functional, GPU matrix multiplication kernel. You can use it like any Python function:

```python
import torch
from solutions.exercise_4 import matmul # Import your completed function

# Create large matrices on the GPU
M, N, K = 2048, 4096, 1024
A = torch.randn(M, K, device='cuda', dtype=torch.float32)
B = torch.randn(K, N, device='cuda', dtype=torch.float32)

# Run your 100% Triton implementation
output_triton = matmul(A, B)

# Compare with PyTorch's highly optimized implementation
output_pytorch = torch.matmul(A, B)

print(f"Max difference: {(output_triton - output_pytorch).abs().max():.2e}")
print("✅ Your kernel produces the correct result!")
```

## 🔧 Troubleshooting

### Common Issues

**CUDA Not Available or `triton.runtime.autotuner.OutOfResources`:**
You can run any exercise in CPU simulation mode for debugging. Simply set the environment variable before running the script.

```bash
# Use CPU simulation mode for debugging exercise 1
export TRITON_INTERPRETER=1
uv run exercises/exercise_1.py
```

**Compilation Errors:**
Ensure your NVIDIA drivers, CUDA toolkit, and PyTorch versions are compatible.
```bash
# Check GPU and driver status
nvidia-smi

# Verify nvcc compiler is available (if CUDA toolkit is installed)
nvcc --version

# Check PyTorch and Triton versions
python -c "import torch; print(torch.__version__)"
python -c "import triton; print(triton.__version__)"
```

## 📚 Learning Resources

### Core Concepts Covered

-   **GPU Execution Model:** Grids, Program IDs, and the Triton vs. CUDA paradigm.
-   **Memory Operations:** Pointers, Offsets, Masking, and safe Load/Store.
-   **Data Layouts:** Handling 1D, 2D, and multi-channel (3D) tensors.
-   **Broadcasting:** Using NumPy-style broadcasting to create coordinate grids.
-   **Tiled Algorithms:** The fundamental pattern for high-performance MatMul.
-   **Kernel Fusion:** The core motivation for using Triton to reduce memory bottlenecks.
-   **Performance Tuning:** A practical introduction to `@triton.autotune`.

### After the Workshop

-   **Triton Puzzles:** Sharpen your skills with this fantastic collection of interactive exercises.
-   **Unsloth Kernels:** Explore production-grade Triton kernels used for optimizing LLMs.
-   **Official Triton Tutorials:** Dive deeper into advanced topics like reductions and scans on the official website.

## 🤝 Community & Support

-   **Issues:** Found a bug or have a suggestion? Please open a GitHub issue.
-   **Discussions:** Have a question about a concept? Start a discussion on GitHub.

---

## 🎉 Ready to Start?

```bash
uv run setup-check
```
