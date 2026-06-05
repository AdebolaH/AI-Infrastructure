# Purpose:
# Run a tiny NCCL-backed distributed AllReduce using PyTorch.
#
# Launch with:
# torchrun --standalone --nproc_per_node=2 nccl_allreduce_hello.py

import os
import socket
import torch
import torch.distributed as dist


def get_env_int(name: str) -> int:
    """Read an integer environment variable created by torchrun."""

    value = os.environ.get(name)

    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")

    return int(value)


def main() -> None:
    # torchrun automatically sets these environment variables.
    # RANK = global process ID across the whole job.
    # WORLD_SIZE = total number of processes/ranks.
    # LOCAL_RANK = process ID on this specific machine.
    rank = get_env_int("RANK")
    world_size = get_env_int("WORLD_SIZE")
    local_rank = get_env_int("LOCAL_RANK")

    hostname = socket.gethostname()

    # Bind this process to the GPU matching its local rank.
    # On one node with 2 GPUs:
    # LOCAL_RANK=0 -> cuda:0
    # LOCAL_RANK=1 -> cuda:1
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Initialise the distributed process group using NCCL.
    # This creates the communication group across all ranks.
    dist.init_process_group(backend="nccl")

    # Create a tensor on this rank's GPU.
    # Rank 0 creates [1.0], rank 1 creates [2.0], etc.
    tensor = torch.tensor([float(rank + 1)], device=device)

    print(
        f"[Before AllReduce] "
        f"host={hostname} rank={rank} local_rank={local_rank} "
        f"world_size={world_size} device={device} tensor={tensor.item()}"
    )

    # Synchronise before the collective for cleaner timing/ordering.
    torch.cuda.synchronize(device)

    # AllReduce SUM:
    # every rank contributes its tensor,
    # NCCL sums the values,
    # every rank receives the same final result.
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Wait until GPU communication has completed.
    torch.cuda.synchronize(device)

    print(
        f"[After AllReduce]  "
        f"host={hostname} rank={rank} local_rank={local_rank} "
        f"world_size={world_size} device={device} tensor={tensor.item()}"
    )

    # Cleanly shut down the process group.
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
