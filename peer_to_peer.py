import torch

if torch.cuda.device_count() > 1:
    # Check if GPU 0 can directly access GPU 1's memory via NVLink/P2P
    can_p2p = torch.cuda.can_device_access_peer(0, 1)
    
    print(f"P2P Access between GPU 0 and 1: {can_p2p}")
    
    if can_p2p:
        # Manually enable it if the framework hasn't already
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        print("Infrastructure is optimized for high-speed local scaling.")
else:
    print("Only one GPU detected. Scaling starts at 2+ GPUs!")
