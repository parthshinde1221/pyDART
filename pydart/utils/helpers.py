import time
import torch
from torch.fx import Node as FxNode

def resolve_arg(arg, node_outputs):
    if isinstance(arg, FxNode):
        return node_outputs.get(arg.name, arg)
    elif isinstance(arg, (list, tuple)):
        return type(arg)(resolve_arg(a, node_outputs) for a in arg)
    elif isinstance(arg, dict):
        return {k: resolve_arg(v, node_outputs) for k, v in arg.items()}
    else:
        return arg

def move_tensor_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True) if obj.device != device else obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_tensor_to_device(x, device) for x in obj)
    elif isinstance(obj, dict):
        return {k: move_tensor_to_device(v, device) for k, v in obj.items()}
    else:
        return obj

def measure_max_transfer_penalty(available_nodes, sample_tensor: torch.Tensor) -> float:
    max_time = 0.0
    for src in available_nodes:
        src_device = torch.device(f"cuda:{src.gpu}") if src.gpu is not None and torch.cuda.is_available() else torch.device("cpu")
        tensor_on_src = sample_tensor.to(src_device)
        for dst in available_nodes:
            dst_device = torch.device(f"cuda:{dst.gpu}") if dst.gpu is not None and torch.cuda.is_available() else torch.device("cpu")
            start = time.time()
            _ = tensor_on_src.to(dst_device)
            if dst_device.type == 'cuda':
                torch.cuda.synchronize(dst_device)
            elapsed = time.time() - start
            max_time = max(max_time, elapsed)
    return max_time

def create_dataloader(batch_size: int, num_samples: int, input_size: tuple):
    from torch.utils.data import DataLoader, TensorDataset
    inputs = torch.randn(num_samples, *input_size)
    targets = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size)
