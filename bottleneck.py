import argparse
import time
from contextlib import nullcontext

import torch
from torch.profiler import (
    profile,
    ProfilerActivity,
    record_function,
)


MATRIX_SIZE = 4096
ITERATIONS = 20


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check nvidia-smi, "
            "your PyTorch CUDA build and GPU visibility."
        )

    return torch.device("cuda:0")


def timed_run(name, workload):
    """
    Measure total wall-clock time for a workload.

    We synchronise before and after because CUDA operations
    execute asynchronously relative to Python.
    """

    torch.cuda.synchronize()

    start = time.perf_counter()

    workload()

    torch.cuda.synchronize()

    end = time.perf_counter()

    elapsed = end - start

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Elapsed: {elapsed:.4f} seconds")
    print(f"Iterations: {ITERATIONS}")
    print(f"Average: {elapsed / ITERATIONS:.6f} seconds/iteration")

    return elapsed


def baseline(device):
    """
    Tensors already live on the GPU.

    This removes repeated CPU tensor creation and host-to-device
    transfers from the measured loop.
    """

    a = torch.randn(
        MATRIX_SIZE,
        MATRIX_SIZE,
        device=device,
    )

    b = torch.randn(
        MATRIX_SIZE,
        MATRIX_SIZE,
        device=device,
    )

    # Warm up CUDA.
    for _ in range(3):
        _ = a @ b

    torch.cuda.synchronize()

    def workload():
        for _ in range(ITERATIONS):
            _ = a @ b

    return timed_run(
        "BASELINE: persistent GPU tensors",
        workload,
    )


def cpu_starved(device):
    """
    Deliberately create data on the CPU every iteration,
    then copy it to the GPU synchronously.

    This represents a workload where the GPU repeatedly waits
    for CPU-side data preparation and transfer.
    """

    def workload():
        for _ in range(ITERATIONS):

            # Allocate and generate data on CPU.
            cpu_a = torch.randn(
                MATRIX_SIZE,
                MATRIX_SIZE,
            )

            cpu_b = torch.randn(
                MATRIX_SIZE,
                MATRIX_SIZE,
            )

            # Synchronous host -> device transfers.
            gpu_a = cpu_a.to(device)
            gpu_b = cpu_b.to(device)

            _ = gpu_a @ gpu_b

    return timed_run(
        "CPU/DATA STARVATION",
        workload,
    )


def sync_heavy(device):
    """
    Force the CPU to wait for the GPU after every operation.

    This deliberately destroys much of CUDA's asynchronous
    execution advantage.
    """

    a = torch.randn(
        MATRIX_SIZE,
        MATRIX_SIZE,
        device=device,
    )

    b = torch.randn(
        MATRIX_SIZE,
        MATRIX_SIZE,
        device=device,
    )

    def workload():
        for _ in range(ITERATIONS):
            _ = a @ b

            # Deliberate synchronisation point.
            torch.cuda.synchronize()

    return timed_run(
        "EXCESSIVE SYNCHRONISATION",
        workload,
    )


def memory_pressure(device):
    """
    Intentionally create significant GPU memory pressure.

    Run this only on a GPU you are authorised to use and not on
    a shared production accelerator.
    """

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)

    gib = 1024 ** 3

    print("\nGPU memory before test")
    print("----------------------")
    print(f"Free:  {free_bytes / gib:.2f} GiB")
    print(f"Total: {total_bytes / gib:.2f} GiB")

    # Allocate about 60% of currently free GPU memory.
    target_bytes = int(free_bytes * 0.60)

    # FP32 = 4 bytes per element.
    elements = target_bytes // 4

    print(
        f"\nAttempting first allocation: "
        f"{target_bytes / gib:.2f} GiB"
    )

    first = torch.empty(
        elements,
        dtype=torch.float32,
        device=device,
    )

    # Touch the allocation so memory really becomes relevant.
    first.fill_(1)

    print(
        f"Allocated: "
        f"{torch.cuda.memory_allocated(device) / gib:.2f} GiB"
    )

    try:
        # Ask for another 60% of the *original* free amount.
        # This should exceed what remains on many systems.
        print(
            f"Attempting second allocation: "
            f"{target_bytes / gib:.2f} GiB"
        )

        second = torch.empty(
            elements,
            dtype=torch.float32,
            device=device,
        )

        second.fill_(2)

        print("Second allocation succeeded.")

    except torch.OutOfMemoryError as error:
        print("\nEXPECTED MEMORY FAILURE")
        print("-----------------------")
        print(type(error).__name__)
        print("GPU memory capacity was exceeded.")

    finally:
        # Release references.
        del first

        if "second" in locals():
            del second

        # Makes unused cached allocations available to other applications.
        torch.cuda.empty_cache()


def profiler_run(device):
    """
    Compare CPU activity with CUDA activity using PyTorch Profiler.
    """

    a = torch.randn(
        MATRIX_SIZE,
        MATRIX_SIZE,
        device=device,
    )

    b = torch.randn(
        MATRIX_SIZE,
        MATRIX_SIZE,
        device=device,
    )

    activities = [
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ]

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as prof:

        with record_function("matrix_workload"):

            for _ in range(5):
                _ = a @ b

    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=15,
        )
    )

    prof.export_chrome_trace(
        "gpu_trace.json"
    )

    print("\nTrace written to gpu_trace.json")


def print_memory(device):
    gib = 1024 ** 3

    print("\nPyTorch GPU memory")
    print("------------------")

    print(
        f"Allocated: "
        f"{torch.cuda.memory_allocated(device) / gib:.3f} GiB"
    )

    print(
        f"Reserved: "
        f"{torch.cuda.memory_reserved(device) / gib:.3f} GiB"
    )

    print(
        f"Peak allocated: "
        f"{torch.cuda.max_memory_allocated(device) / gib:.3f} GiB"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        choices=[
            "baseline",
            "cpu-starved",
            "sync-heavy",
            "memory-pressure",
            "profile",
        ],
    )

    args = parser.parse_args()

    device = require_cuda()

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Mode: {args.mode}")

    torch.cuda.reset_peak_memory_stats(device)

    if args.mode == "baseline":
        baseline(device)

    elif args.mode == "cpu-starved":
        cpu_starved(device)

    elif args.mode == "sync-heavy":
        sync_heavy(device)

    elif args.mode == "memory-pressure":
        memory_pressure(device)

    elif args.mode == "profile":
        profiler_run(device)

    print_memory(device)


if __name__ == "__main__":
    main()
