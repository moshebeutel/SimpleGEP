from typing import Tuple

import torch

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    a = a.mean(0, keepdim=False)
    b = b.mean(0, keepdim=False)
    cosine = torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
    angle_rad = torch.acos(cosine)
    angle_deg = float(angle_rad * 180 / torch.pi)
    return float(cosine), angle_deg


def check_approx_error(L: torch.Tensor, target: torch.Tensor, return_cosine=False) -> float or Tuple[float, float, float]:
    encode = torch.matmul(target, L)  # n x k
    decode = torch.matmul(encode, L.T)
    error = float(torch.sum(torch.square(target - decode)))
    target_sum_squares = float(torch.sum(torch.square(target)))
    assert target_sum_squares > 0, f'Expected positive target_sum_squares. Got {target_sum_squares}'
    if return_cosine:
        cosine, angle_deg = cosine_similarity(target, decode)
        return error / target_sum_squares, cosine, angle_deg

    return error / target_sum_squares

def flatten_tensor(tensor_list: list[ torch.Tensor]) -> torch.Tensor:
    for i in range(len(tensor_list)):
        tensor_list[i] = tensor_list[i].reshape([tensor_list[i].shape[0], -1])
    flatten_param = torch.cat(tensor_list, dim=1)
    del tensor_list
    return flatten_param

def clip_column(tsr: torch.Tensor, clip: float=1.0, inplace: bool=False) -> torch.Tensor or None:
    if inplace:
        inplace_clipping(tsr, torch.tensor(clip).cuda())
        return None
    else:
        norms = torch.norm(tsr, dim=1)

        scale = torch.clamp(clip / norms, max=1.0)
        return tsr * scale.view(-1, 1)


@torch.jit.script
def inplace_clipping(matrix: torch.Tensor, clip: float) -> None:
    n, m = matrix.shape
    for i in range(n):
        # Normalize the i'th row
        col = matrix[i:i + 1, :]
        col_norm = torch.sqrt(torch.sum(col ** 2))
        if col_norm > clip:
            col /= (col_norm / clip)

def clip_to_value(tensor, clip_value):
    return clip_column(tensor, clip=clip_value, inplace=False)

def clip_to_max_norm(tensor, clip_value):
    max_norm = torch.norm(tensor, dim=1).max().item()
    clip_value = min(clip_value, max_norm)
    return clip_column(tensor, clip=clip_value, inplace=False)

def clip_to_median_norm(tensor, clip_value):
    median_norm = torch.median(torch.norm(tensor, dim=1)).item()
    clip_value = min(clip_value, median_norm)
    return clip_column(tensor, clip=clip_value, inplace=False)

