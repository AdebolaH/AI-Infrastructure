# gpu_pair_test.py
# Purpose:
# Run a simple workload on the GPUs visible to this process.
#
# You control visible GPUs using CUDA_VISIBLE_DEVICES.
#
# Example:
# CUDA_VISIBLE_DEVICES=0,1 python gpu_pair_test.py
# CUDA_VISIBLE_DEVICES=0,3 python gpu_pair_test.py

import os
import time
import torch


def show_visible_gpus() -> None:
    """Print the GPUs visible to PyTorch."""

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("PyTorch cannot see CUDA. Stop here and fix GPU access first.")
        return

    print(f"Visible GPU count from PyTorch: {torch.cuda.device_count()}")

    for local_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(local_id)

        # local_id is the CUDA index seen by this process.
        # It may not match the physical GPU ID if CUDA_VISIBLE_DEVICES is set.
        print(
            f"Local cuda:{local_id} -> {props.name}, "
            f"{props.total_memory / 1024**3:.2f} GB"
        )


def run_single_gpu_compute(local_gpu_id: int, matrix_size: int = 8192) -> float:
    """Run matrix multiplication on one visible GPU and return elapsed time."""

    device = torch.device(f"cuda:{local_gpu_id}")

    # Allocate tensors directly on the target GPU.
    a = torch.randn((matrix_size, matrix_size), device=device)
    b = torch.randn((matrix_size, matrix_size), device=device)

    # Warm up once so the first measured run is less noisy.
    _ = a @ b
    torch.cuda.synchronize(device)

    start = time.perf_counter()

    # Real GPU compute.
    c = a @ b

    # GPU work is asynchronous, so wait before stopping the timer.
    torch.cuda.synchronize(device)

    end = time.perf_counter()

    # Keep c alive until after synchronisation.
    _ = c.shape

    return end - start


def main() -> None:
    show_visible_gpus()

    if not torch.cuda.is_available():
        return

    if torch.cuda.device_count() < 2:
        print("\nThis lesson needs at least 2 visible GPUs.")
        print("Try exposing two GPUs with CUDA_VISIBLE_DEVICES=0,1")
        return

    print("\nRunning compute on each visible GPU...")

    for local_gpu_id in range(torch.cuda.device_count()):
        elapsed = run_single_gpu_compute(local_gpu_id)
        print(f"cuda:{local_gpu_id} compute time: {elapsed:.4f} seconds")


if __name__ == "__main__":
    main()
