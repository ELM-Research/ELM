import torch

def batch_to_device(v, device):
    if isinstance(v, torch.Tensor):
        return v.to(device)
    if isinstance(v, dict):
        return {k: batch_to_device(x, device) for k, x in v.items()}
    return v
