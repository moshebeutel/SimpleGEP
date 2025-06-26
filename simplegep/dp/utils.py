import torch


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

