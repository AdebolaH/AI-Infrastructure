# This script proves that PyTorch can:
# 1. detect the GPU,
# 2. allocate GPU memory,
# 3. run real GPU compute,
# 4. report GPU memory usage.

import time
import torch


def print_gpu_info() -> None:
    """Print basic information about all visible CUDA GPUs."""

    # torch.cuda.is_available() confirms whether PyTorch can use CUDA.
    if not torch.cuda.is_available():
        print("CUDA is not available to PyTorch.")
        print("Fix the driver, CUDA/PyTorch package, or container GPU access first.")
        return

    # Count visible GPUs.
    gpu_count = torch.cuda.device_count()
    print(f"Visible CUDA GPUs: {gpu_count}")

    # Print information for each GPU.
    for gpu_id in range(gpu_count):
        props = torch.cuda.get_device_properties(gpu_id)

        print(f"\nGPU {gpu_id}")
        print(f"Name: {props.name}")
        print(f"Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"CUDA capability: {props.major}.{props.minor}")


def run_gpu_compute() -> None:
    """Run a real matrix multiplication on GPU 0."""

    if not torch.cuda.is_available():
        return

    # Select GPU 0.
    device = torch.device("cuda:0")

    # Empty any cached memory from previous runs.
    # This does not free memory used by active tensors.
    torch.cuda.empty_cache()

    print("\nBefore allocation:")
    print_memory(device)

    # Create two large matrices directly on the GPU.
    # These tensors live in GPU memory, not normal system RAM.
    a = torch.randn((8192, 8192), device=device)
    b = torch.randn((8192, 8192), device=device)

    print("\nAfter allocation:")
    print_memory(device)

    # GPU operations are asynchronous.
    # Synchronise before timing so previous GPU work is complete.
    torch.cuda.synchronize()

    start = time.perf_counter()

    # This is real GPU compute.
    c = a @ b

    # Synchronise after the operation so timing reflects completed GPU work.
    torch.cuda.synchronize()

    end = time.perf_counter()

    print("\nAfter matrix multiplication:")
    print_memory(device)

    print(f"\nOutput tensor shape: {tuple(c.shape)}")
    print(f"GPU compute time: {end - start:.4f} seconds")

    # Keep c alive until after memory reporting.
    # If tensors are deleted, PyTorch may release/reuse memory.


def print_memory(device: torch.device) -> None:
    """Print PyTorch CUDA memory statistics for one device."""

    # memory_allocated = memory currently occupied by tensors.
    allocated_gb = torch.cuda.memory_allocated(device) / 1024**3

    # memory_reserved = memory reserved by PyTorch's caching allocator.
    reserved_gb = torch.cuda.memory_reserved(device) / 1024**3

    print(f"PyTorch allocated memory: {allocated_gb:.2f} GB")
    print(f"PyTorch reserved memory:  {reserved_gb:.2f} GB")


if __name__ == "__main__":
    print_gpu_info()
    run_gpu_compute()
