from dataclasses import dataclass
from typing import Optional

from simplegep.dp.rdp_accountant import get_sigma


@dataclass
class DPParams:
    delta: float
    epsilon: float
    sigma: float
    sampling_prob: float
    steps: int



def get_dp_params(batchsize: int, num_training_samples: int, num_epochs: int, epsilon: float, sigma: Optional[float] = None) -> DPParams:
    sampling_prob: float = batchsize / num_training_samples
    steps: int = int(num_epochs / sampling_prob)
    delta: float = 1 / num_training_samples
    sigma, eps = get_sigma(sampling_prob, steps, epsilon, delta, rgp=False) if epsilon > 0.0 else (sigma, epsilon)
    assert eps > 0.0 or sigma is not None, f'Expected a predefined noise multiplier or epsilon > 0.0. Got epsilon={epsilon} and sigma={sigma}'
    return DPParams(
        delta=delta,
        epsilon=eps,
        sigma=sigma,
        sampling_prob=sampling_prob,
        steps=steps
    )
