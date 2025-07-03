import torch
from simplegep.dp.utils import clip_to_max_norm, clip_to_median_norm, clip_to_value

clipping_strategy_hub = {
    'value': clip_to_value,
    'median': clip_to_median_norm,
    'max': clip_to_max_norm
}

def get_noise(sigma, batchsize, shape, device):
    noise = torch.torch.normal(0, sigma / batchsize, size=shape, device=device)
    return noise

class GradsProcessor:
    def __init__(self, clip_strategy_name: str, noise_multiplier, clip_value: float):
        self.clip_strategy_name = clip_strategy_name
        self.noise_multiplier_list = [noise_multiplier] if isinstance(noise_multiplier, float) else noise_multiplier
        self.clip_value = clip_value
        self.clip_func = clipping_strategy_hub[clip_strategy_name]
        self._epoch = 0
    def get_noise_multiplier(self, epoch: int) -> float:
        return self.noise_multiplier_list[epoch] if epoch < len(self.noise_multiplier_list) else self.noise_multiplier_list[-1]

    def process_grads(self, grads: torch.Tensor) -> torch.Tensor:
        clipped_grads = self.clip_func(grads, self.clip_value)
        clipped_norm = torch.linalg.norm(clipped_grads, dim=1).max().item()
        mean_grad = clipped_grads.mean(dim=0).squeeze()
        sigma = self.get_noise_multiplier(self._epoch) * clipped_norm
        noise = get_noise(sigma, grads.shape[0], mean_grad.shape, grads.device)
        noised_grads = mean_grad + noise
        self._epoch += 1
        return noised_grads


