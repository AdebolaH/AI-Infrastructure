# real_gpu_probe.py

# This script checks whether PyTorch can use your NVIDIA GPU.
# It also runs a small matrix multiplication on the GPU.

import torch
import time


def main():
    # Check whether CUDA is available to PyTorch.
    # If this is False, PyTorch cannot currently use your NVIDIA GPU.
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if not cuda_available:
        print("PyTorch cannot see a CUDA GPU.")
        print("Check NVIDIA driver, CUDA/PyTorch install, or container GPU access.")
        return

    # Count how many GPUs PyTorch can see.
    gpu_count = torch.cuda.device_count()
    print(f"GPU count: {gpu_count}")

    # Print basic information about each GPU.
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)

        print(f"\nGPU {i}")
        print(f"Name: {props.name}")
        print(f"Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"CUDA capability: {props.major}.{props.minor}")

    # Select GPU 0.
    device = torch.device("cuda:0")

    # Create two large random matrices directly on the GPU.
    # This forces real GPU memory allocation and compute.
    a = torch.randn((4096, 4096), device=device)
    b = torch.randn((4096, 4096), device=device)

    # Synchronise before timing.
    # GPU work is asynchronous, so timing without sync can be misleading.
    torch.cuda.synchronize()

    start = time.time()

    # Matrix multiplication runs on the GPU.
    c = a @ b

    # Synchronise again so Python waits until the GPU work is finished.
    torch.cuda.synchronize()

    end = time.time()

    print(f"\nMatrix multiplication completed.")
    print(f"Output tensor shape: {c.shape}")
    print(f"Time taken: {end - start:.4f} seconds")

    # Show memory currently allocated by PyTorch on GPU 0.
    allocated_gb = torch.cuda.memory_allocated(device) / 1024**3
    reserved_gb = torch.cuda.memory_reserved(device) / 1024**3

    print(f"GPU memory allocated by PyTorch: {allocated_gb:.2f} GB")
    print(f"GPU memory reserved by PyTorch: {reserved_gb:.2f} GB")


if __name__ == "__main__":
    main()
