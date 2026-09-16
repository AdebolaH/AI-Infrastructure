import time
import statistics

import torch


MATRIX_SIZE = 8192
WARMUP_RUNS = 3
BENCHMARK_RUNS = 10


def benchmark_cpu(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    timings = []

    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()

        _ = a @ b

        end = time.perf_counter()

        timings.append(end - start)

    return timings


def benchmark_gpu(
    a: torch.Tensor,
    b: torch.Tensor,
) -> list[float]:
    timings = []

    # Warm-up runs are intentionally not measured.
    #
    # The first CUDA operations can contain one-off setup overhead
    # that would distort our benchmark.
    for _ in range(WARMUP_RUNS):
        _ = a @ b

    torch.cuda.synchronize()

    for _ in range(BENCHMARK_RUNS):

        # CUDA events measure GPU execution time rather than relying
        # purely on the CPU's wall clock.
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        _ = a @ b

        end_event.record()

        # CUDA work is asynchronous.
        # Wait until GPU work has actually completed before reading time.
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)

        timings.append(elapsed_ms / 1000)

    return timings


def summarize(name: str, timings: list[float]) -> None:
    print(f"\n{name}")
    print("-" * len(name))

    print(f"Runs: {len(timings)}")
    print(f"Mean:   {statistics.mean(timings):.6f} seconds")
    print(f"Median: {statistics.median(timings):.6f} seconds")
    print(f"Min:    {min(timings):.6f} seconds")
    print(f"Max:    {max(timings):.6f} seconds")


def main():
    print(f"Matrix size: {MATRIX_SIZE} x {MATRIX_SIZE}")

    #
    # CPU benchmark
    #

    print("\nCreating CPU tensors...")

    cpu_a = torch.randn(MATRIX_SIZE, MATRIX_SIZE)
    cpu_b = torch.randn(MATRIX_SIZE, MATRIX_SIZE)

    cpu_times = benchmark_cpu(cpu_a, cpu_b)

    summarize("CPU results", cpu_times)

    #
    # GPU benchmark
    #

    if not torch.cuda.is_available():
        print("\nCUDA unavailable. GPU benchmark skipped.")
        return

    device = torch.device("cuda:0")

    print(f"\nRunning GPU benchmark on {torch.cuda.get_device_name(device)}")

    # Reset our peak measurement before the benchmark.
    torch.cuda.reset_peak_memory_stats(device)

    gpu_a = torch.randn(
        MATRIX_SIZE,
        MATRIX_SIZE,
        device=device,
    )

    gpu_b = torch.randn(
        MATRIX_SIZE,
        MATRIX_SIZE,
        device=device,
    )

    gpu_times = benchmark_gpu(gpu_a, gpu_b)

    summarize("GPU results", gpu_times)

    #
    # Memory measurements
    #

    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3

    print("\nGPU memory")
    print("----------")
    print(f"Currently allocated: {allocated:.3f} GiB")
    print(f"Currently reserved:  {reserved:.3f} GiB")
    print(f"Peak allocated:      {peak_allocated:.3f} GiB")

    #
    # Compute speed-up
    #

    cpu_median = statistics.median(cpu_times)
    gpu_median = statistics.median(gpu_times)

    speedup = cpu_median / gpu_median

    print("\nComparison")
    print("----------")
    print(f"GPU speed-up over CPU: {speedup:.2f}x")


if __name__ == "__main__":
    main()
