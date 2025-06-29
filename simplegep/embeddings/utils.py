import torch


def normalize_return_transform(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    center = x.mean(dim=0)
    scale = x.std(dim=0)
    normalized_x = (x - center) / scale
    return normalized_x, center, scale