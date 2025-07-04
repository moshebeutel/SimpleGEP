import math

import numpy as np
from tqdm import trange

from simplegep.dp.rdp_accountant import compute_rdp, get_privacy_spent, get_sigma


def get_epsilon_from_epsilon_bar(epsilon_bar: float, alpha:float, delta: float):
    return epsilon_bar - math.log(delta)/(alpha-1)

def get_epsilon_bar_from_epsilon(epsilon: float, alpha:float, delta: float):
    return epsilon + math.log(delta)/(alpha-1)

def privacy_budget_left(sampling_prob, steps, cur_sigma, delta, rdp_orders=32):
    orders = np.arange(2, rdp_orders, 2.0)
    rdp = compute_rdp(sampling_prob, cur_sigma, steps, orders)
    cur_eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
    # print(f"Current eps: {cur_eps} optimal order {opt_order} current sigma {cur_sigma} rdp {rdp}")
    epsilon_bar = get_epsilon_bar_from_epsilon(cur_eps, opt_order, delta)
    # print(f"epsilon_bar: {epsilon_bar}")
    return float(cur_eps), epsilon_bar

def get_varying_sigma_values(q, n_epoch, eps, delta, initial_sigma_factor, sigma_factor_decrease_factor):
    accumulated_epsilon_bar = 0
    sigma_factor = initial_sigma_factor
    sigmas = []
    pbar = trange(n_epoch)
    for i in pbar:

        steps_in_epoch = int(1 / q)
        sigma, previous_eps = get_sigma(q=q, T=steps_in_epoch, eps=eps, delta=delta)
        sigma *= sigma_factor
        sigmas.append(sigma)
        sigma_factor *= sigma_factor_decrease_factor
        epsilon, epsilon_bar = privacy_budget_left(q, steps_in_epoch, sigma, delta)
        accumulated_epsilon_bar += epsilon_bar
        accumulated_epsilon = get_epsilon_from_epsilon_bar(accumulated_epsilon_bar, 32, delta)
        pbar.set_description(f"accumulated epsilon: {accumulated_epsilon} accumulated epsilon bar: {accumulated_epsilon_bar}")
        if accumulated_epsilon > eps:
            sigmas = sigmas[:-1]
            break

    return sigmas

if __name__ == "__main__":
    batchsize = 128
    n_training = 50000
    n_epoch = 25
    delta = 1 / n_epoch
    epsilon = 8
    sampling_prob = batchsize / n_training
    steps = int(n_epoch / sampling_prob)
    sigmas = get_varying_sigma_values(sampling_prob, n_epoch, epsilon, delta,
                                      10.0, 0.89)

    print(sigmas)
    print(f"Number of sigmas: {len(sigmas)}")
    print(f'First sigma: {sigmas[0]}')
    print(f"Final sigma: {sigmas[-1]}")