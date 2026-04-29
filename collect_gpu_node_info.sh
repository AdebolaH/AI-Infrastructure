#!/usr/bin/env bash

# collect_gpu_node_info.sh
# This script collects basic evidence from a real GPU node.
# It helps you understand GPU visibility, topology, CPU layout and NUMA layout.

set -euo pipefail

# Create an output directory with a timestamp.
OUTPUT_DIR="gpu_node_report_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Collecting GPU node information into: $OUTPUT_DIR"

# Check whether NVIDIA GPUs are visible.
# This captures GPU model, driver version, CUDA compatibility, memory and utilisation.
nvidia-smi > "$OUTPUT_DIR/nvidia_smi.txt"

# Capture GPU-to-GPU, GPU-to-NIC, CPU and NUMA affinity.
nvidia-smi topo -m > "$OUTPUT_DIR/nvidia_smi_topo_m.txt"

# Capture PCI-only topology.
# This helps separate PCIe paths from NVLink paths.
nvidia-smi topo -mp > "$OUTPUT_DIR/nvidia_smi_topo_mp.txt"

# Capture CPU architecture, sockets, cores and threads.
lscpu > "$OUTPUT_DIR/lscpu.txt"

# Capture NUMA node layout.
numactl --hardware > "$OUTPUT_DIR/numactl_hardware.txt"

# Capture NVIDIA PCI devices.
lspci | grep -i nvidia > "$OUTPUT_DIR/lspci_nvidia.txt" || true

echo "Done."
echo "Review these files:"
ls -1 "$OUTPUT_DIR"
