# parse_gpu_topology.py
# Purpose:
# Read nvidia-smi topo -m output and identify useful GPU relationships.
#
# Usage:
# python parse_gpu_topology.py gpu_topology_report_xxxxx/02_nvidia_smi_topo_m.txt

import sys
from pathlib import Path


# Rank connection types from strongest to weakest.
# This is simplified, but useful for learning.
CONNECTION_SCORE = {
    "X": 0,
    "NV": 1,
    "PIX": 2,
    "PXB": 3,
    "PHB": 4,
    "NODE": 5,
    "SYS": 6,
}


def normalise_link(value: str) -> str:
    """Convert values like NV4 into NV, while keeping PIX/PXB/PHB/NODE/SYS."""

    value = value.strip()

    # NVLink entries often look like NV1, NV2, NV4, etc.
    if value.startswith("NV"):
        return "NV"

    return value


def score_link(value: str) -> int:
    """Return a numeric score for a topology link. Lower is better."""

    link = normalise_link(value)
    return CONNECTION_SCORE.get(link, 99)


def parse_topology(file_path: Path):
    """Parse a simple nvidia-smi topo -m table."""

    lines = file_path.read_text().splitlines()

    # Find the header line that starts with GPU columns.
    header_line = None
    for line in lines:
        if "GPU0" in line and "CPU Affinity" in line:
            header_line = line
            break

    if header_line is None:
        raise ValueError("Could not find GPU topology header line.")

    headers = header_line.split()

    # Keep only GPU column names from the header.
    gpu_headers = [h for h in headers if h.startswith("GPU")]

    relationships = []

    for line in lines:
        parts = line.split()

        # Data rows usually start with GPU0, GPU1, etc.
        if not parts or not parts[0].startswith("GPU"):
            continue

        row_gpu = parts[0]

        # The next values correspond to GPU columns.
        links = parts[1 : 1 + len(gpu_headers)]

        for col_gpu, link in zip(gpu_headers, links):
            # Avoid self-links and duplicates.
            if row_gpu == col_gpu:
                continue

            # Only keep one direction, e.g. GPU0-GPU1, not GPU1-GPU0 again.
            if row_gpu < col_gpu:
                relationships.append((row_gpu, col_gpu, link, score_link(link)))

    return relationships


def main():
    if len(sys.argv) != 2:
        print("Usage: python parse_gpu_topology.py <nvidia_smi_topo_m.txt>")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    relationships = parse_topology(file_path)

    if not relationships:
        print("No GPU-to-GPU relationships found.")
        return

    # Sort links from best to worst.
    relationships.sort(key=lambda item: item[3])

    print("GPU relationships from best to worst:")
    print()

    for gpu_a, gpu_b, link, score in relationships:
        print(f"{gpu_a} <-> {gpu_b}: {link}")

    print()
    print("Best pair:")
    print(f"{relationships[0][0]} <-> {relationships[0][1]} via {relationships[0][2]}")

    print()
    print("Worst pair:")
    print(f"{relationships[-1][0]} <-> {relationships[-1][1]} via {relationships[-1][2]}")


if __name__ == "__main__":
    main()
