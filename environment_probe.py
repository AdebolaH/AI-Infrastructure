import platform
import torch


def main():
    print("=== System ===")
    print(f"OS: {platform.platform()}")
    print(f"Python/PyTorch: {torch.__version__}")

    print("\n=== CUDA ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("\nNo usable CUDA GPU detected.")
        return

    gpu_count = torch.cuda.device_count()

    print(f"Visible GPU count: {gpu_count}")

    for gpu_id in range(gpu_count):
        props = torch.cuda.get_device_properties(gpu_id)

        print(f"\nGPU {gpu_id}")
        print(f"Name: {props.name}")
        print(f"Total memory: {props.total_memory / 1024**3:.2f} GiB")
        print(f"Compute capability: {props.major}.{props.minor}")


if __name__ == "__main__":
    main()
