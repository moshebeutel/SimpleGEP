from dataclasses import dataclass

from simplegep.dp.rdp_accountant import get_sigma


@dataclass
class DPParams:
    delta: float
    epsilon: float
    sigma: float


def get_dp_params(batchsize: int, num_training_samples: int, num_epochs: int, epsilon: float) -> DPParams:
    sampling_prob: float = batchsize / num_training_samples
    steps: int = int(num_epochs / sampling_prob)
    delta: float = 1 / num_training_samples
    sigma, eps = get_sigma(sampling_prob, steps, epsilon, delta, rgp=False)

    return DPParams(
        delta=delta,
        epsilon=eps,
        sigma=sigma
    )
