"""
Demonstrate the memory wall: same FLOPs, different arithmetic intensity,
dramatically different throughput.

Kernel A: low intensity (~1 FLOP/byte) — memory-bound
Kernel B: high intensity (~150+ FLOPs/byte) — compute-bound (matmul)
"""
import time
import torch

device = torch.device("cuda:0")
torch.cuda.set_device(device)


def bench_kernel(label, fn, flops, iters=50):
    """Run fn iters times, report TFLOPS achieved."""
    for _ in range(5):  # warm-up
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tflops = (flops * iters) / elapsed / 1e12
    print(f"{label:45s} {tflops:7.2f} TFLOPS")
    return tflops


# Kernel A: elementwise — low arithmetic intensity
N = 256 * 1024 * 1024  # 256M elements = 512 MB in fp16
a = torch.randn(N, device=device, dtype=torch.float16)
b = torch.randn(N, device=device, dtype=torch.float16)
c = torch.empty(N, device=device, dtype=torch.float16)

def elementwise():
    torch.add(a, b, out=c)

# 1 FLOP per element (just the add)
flops_elementwise = N * 1
bench_kernel("Elementwise add (1 FLOP/byte)", elementwise, flops_elementwise)


# Kernel B: matmul — high arithmetic intensity, hits Tensor Cores
M = 8192
x = torch.randn(M, M, device=device, dtype=torch.float16)
y = torch.randn(M, M, device=device, dtype=torch.float16)

def matmul():
    torch.matmul(x, y)

# Matmul is 2 * M^3 FLOPs
flops_matmul = 2 * M ** 3
bench_kernel(f"Matmul {M}x{M} fp16 (Tensor Cores)", matmul, flops_matmul)


# Theoretical ceilings on RTX 3090
print("\nFor reference (RTX 3090):")
print("  HBM bandwidth ceiling:    ~936 GB/s")
print("  FP16 Tensor Core ceiling: ~142 TFLOPS")
