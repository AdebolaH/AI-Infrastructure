# nccl_allreduce_debug.py
# Purpose:
# Run a timed NCCL-backed AllReduce using PyTorch Distributed.
#
# Launch:
# CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 nccl_allreduce_debug.py
#
# With debug:
# NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET \
# CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 nccl_allreduce_debug.py

import os
import time
import socket
import torch
import torch.distributed as dist


def env_int(name: str) -> int:
    """Read an integer environment variable created by torchrun."""
    value = os.environ.get(name)

    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")

    return int(value)


def main() -> None:
    # torchrun sets these automatically.
    rank = env_int("RANK")
    world_size = env_int("WORLD_SIZE")
    local_rank = env_int("LOCAL_RANK")

    hostname = socket.gethostname()

    # Each local process gets one visible GPU.
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Initialise NCCL process group.
    dist.init_process_group(backend="nccl")

    # 256 * 1024 * 1024 float32 values ≈ 1 GB tensor.
    tensor_size = 256 * 1024 * 1024

    # Each rank creates a tensor filled with its rank-specific value.
    # Rank 0 creates all 1s, rank 1 creates all 2s, etc.
    tensor = torch.full(
        size=(tensor_size,),
        fill_value=float(rank + 1),
        dtype=torch.float32,
        device=device,
    )

    print(
        f"[Rank info] host={hostname} rank={rank} "
        f"local_rank={local_rank} world_size={world_size} device={device}"
    )

    # Barrier ensures all ranks reach this point before timing.
    dist.barrier()

    # Warm-up AllReduce.
    # First communication can include setup overhead, so do not measure it.
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)

    # Reset tensor to rank-specific values before measured run.
    tensor.fill_(float(rank + 1))
    torch.cuda.synchronize(device)

    # Time the measured AllReduce.
    start = time.perf_counter()

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    torch.cuda.synchronize(device)
    end = time.perf_counter()

    elapsed = end - start

    # Check first value only.
    # For 2 ranks, expected result = 1 + 2 = 3.
    # For 4 ranks, expected result = 1 + 2 + 3 + 4 = 10.
    first_value = tensor[0].item()
    expected = sum(float(i + 1) for i in range(world_size))

    print(
        f"[Result] rank={rank} elapsed={elapsed:.6f}s "
        f"first_value={first_value} expected={expected}"
    )

    if first_value != expected:
        print(f"[ERROR] rank={rank} AllReduce result is incorrect.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
