import torch
import time

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Initialize large vectors on the CPU (Host)
size = 100_000_000
a_cpu = torch.randn(size)
b_cpu = torch.randn(size)

# 2. Transfer data to GPU (Device) - This is the "Data Transfer" bottleneck
start_transfer = time.time()
a_gpu = a_cpu.to(device)
b_gpu = b_cpu.to(device)
print(f"Transfer Time: {time.time() - start_transfer:.4f}s")

# 3. Launch the Kernel (Vector Addition)
# PyTorch handles the Grid/Block/Thread layout for you here
start_compute = time.time()
c_gpu = a_gpu + b_gpu
torch.cuda.synchronize() # Wait for GPU to finish
print(f"GPU Compute Time: {time.time() - start_compute:.4f}s")
