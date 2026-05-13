#!/usr/bin/env bash

# gpu_topology_inspector.sh
# Purpose:
# Collect useful evidence from a real GPU node.
# This helps you understand CPU, NUMA, PCIe and GPU topology.

set -euo pipefail

# Create a timestamped report directory.
REPORT_DIR="gpu_topology_report_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPORT_DIR"

echo "Creating GPU topology report in: $REPORT_DIR"
echo

# -------------------------------
# 1. Basic GPU visibility
# -------------------------------

echo "[1/8] Collecting nvidia-smi output..."

# Shows GPU model, memory, utilisation, driver and CUDA compatibility.
nvidia-smi > "$REPORT_DIR/01_nvidia_smi.txt" 2>&1 || {
  echo "nvidia-smi failed. GPU driver or GPU visibility may be broken."
  exit 1
}

# -------------------------------
# 2. GPU topology matrix
# -------------------------------

echo "[2/8] Collecting GPU topology matrix..."

# Shows GPU-to-GPU, GPU-to-NIC, CPU affinity and NUMA affinity.
nvidia-smi topo -m > "$REPORT_DIR/02_nvidia_smi_topo_m.txt" 2>&1 || true

# Shows PCIe-only topology where supported.
nvidia-smi topo -mp > "$REPORT_DIR/03_nvidia_smi_topo_mp.txt" 2>&1 || true

# -------------------------------
# 3. CPU architecture
# -------------------------------

echo "[3/8] Collecting CPU architecture..."

# Shows sockets, cores, threads, NUMA nodes and CPU model.
lscpu > "$REPORT_DIR/04_lscpu.txt" 2>&1 || true

# -------------------------------
# 4. NUMA hardware layout
# -------------------------------

echo "[4/8] Collecting NUMA layout..."

# Shows NUMA nodes, CPUs attached to each NUMA node and node memory.
numactl --hardware > "$REPORT_DIR/05_numactl_hardware.txt" 2>&1 || true

# -------------------------------
# 5. PCIe NVIDIA devices
# -------------------------------

echo "[5/8] Collecting NVIDIA PCIe devices..."

# Shows NVIDIA devices at PCIe level.
lspci | grep -i nvidia > "$REPORT_DIR/06_lspci_nvidia.txt" 2>&1 || true

# -------------------------------
# 6. Network adapters
# -------------------------------

echo "[6/8] Collecting network adapter evidence..."

# Shows Mellanox/NVIDIA network adapters if present.
lspci | grep -Ei "mellanox|nvidia.*ethernet|infiniband|network" > "$REPORT_DIR/07_lspci_network.txt" 2>&1 || true

# -------------------------------
# 7. RDMA / InfiniBand evidence
# -------------------------------

echo "[7/8] Collecting RDMA/InfiniBand evidence..."

# Shows RDMA devices if rdma-core tools are installed.
if command -v rdma >/dev/null 2>&1; then
  rdma link show > "$REPORT_DIR/08_rdma_link_show.txt" 2>&1 || true
else
  echo "rdma command not installed." > "$REPORT_DIR/08_rdma_link_show.txt"
fi

# Shows InfiniBand verbs devices if installed.
if command -v ibv_devinfo >/dev/null 2>&1; then
  ibv_devinfo > "$REPORT_DIR/09_ibv_devinfo.txt" 2>&1 || true
else
  echo "ibv_devinfo command not installed." > "$REPORT_DIR/09_ibv_devinfo.txt"
fi

# -------------------------------
# 8. Python/PyTorch CUDA check
# -------------------------------

echo "[8/8] Collecting PyTorch CUDA evidence..."

python - <<'PY' > "$REPORT_DIR/10_pytorch_cuda_check.txt" 2>&1
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Visible CUDA GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for gpu_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(gpu_id)
        print()
        print(f"GPU {gpu_id}")
        print(f"Name: {props.name}")
        print(f"Total memory GB: {props.total_memory / 1024**3:.2f}")
        print(f"Compute capability: {props.major}.{props.minor}")
PY

echo
echo "Report created successfully:"
echo "$REPORT_DIR"
echo
echo "Files collected:"
ls -1 "$REPORT_DIR"
