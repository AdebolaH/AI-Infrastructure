# Measure real bandwidth: CPU<->GPU (PCIe), GPU<->GPU (NVLink), and intra-GPU (HBM).

import time
import torch

assert torch.cuda.is_available(), "CUDA not available"
assert torch.cuda.device_count() >= 2, "Need 2 GPUs for this test"

SIZE_MB = 512  # 512 MB transfer per test
N = SIZE_MB * 1024 * 1024 // 4  # number of float32 elements
ITERS = 20  # repeat to average out noise


def bench(label, fn):
    """Run fn ITERS times, return GB/s."""
    # Warm-up — first call is always slow (allocator, kernel launch, etc.)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    bytes_moved = SIZE_MB * 1024 * 1024 * ITERS
    gbps = bytes_moved / elapsed / 1e9
    print(f"{label:35s} {gbps:7.1f} GB/s")
    return gbps


def main():
    # Allocate buffers
    cpu_tensor = torch.randn(N, dtype=torch.float32, pin_memory=True)
    gpu0_tensor = torch.empty(N, dtype=torch.float32, device="cuda:0")
    gpu0_tensor_b = torch.empty(N, dtype=torch.float32, device="cuda:0")
    gpu1_tensor = torch.empty(N, dtype=torch.float32, device="cuda:1")

    print(f"\nTransfer size: {SIZE_MB} MB, iterations: {ITERS}\n")

    # 1. Host -> Device (PCIe)
    bench("CPU -> GPU0 (PCIe)",
          lambda: gpu0_tensor.copy_(cpu_tensor, non_blocking=True))

    # 2. Device -> Host (PCIe)
    bench("GPU0 -> CPU (PCIe)",
          lambda: cpu_tensor.copy_(gpu0_tensor, non_blocking=True))

    # 3. GPU0 -> GPU1 (NVLink if bridge present, else PCIe peer-to-peer)
    bench("GPU0 -> GPU1 (NVLink/PCIe)",
          lambda: gpu1_tensor.copy_(gpu0_tensor, non_blocking=True))

    # 4. Intra-GPU HBM (this is your HBM bandwidth ceiling)
    bench("GPU0 -> GPU0 (HBM)",
          lambda: gpu0_tensor_b.copy_(gpu0_tensor, non_blocking=True))


if __name__ == "__main__":
    main()
