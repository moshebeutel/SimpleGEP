import math
from functools import partial
from typing import Sequence

import numpy as np
from tqdm import trange, tqdm
from simplegep.dp.rdp_accountant import compute_rdp, get_privacy_spent, get_sigma


def get_epsilon_from_epsilon_bar(epsilon_bar: float, alpha: float, delta: float):
    """
    Convert Rényi DP budget (epsilon_bar) to approximate DP epsilon.

    Parameters:
        epsilon_bar (float): Rényi DP budget at order alpha (a.k.a. epsilon-bar).
        alpha (float): Rényi order (> 1).
        delta (float): Target delta for (epsilon, delta)-DP.

    Returns:
        float: The corresponding (approximate) DP epsilon.
    """
    return epsilon_bar - math.log(delta) / (alpha - 1)


def get_epsilon_bar_from_epsilon(epsilon: float, alpha: float, delta: float):
    """
    Convert approximate DP epsilon to Rényi DP budget (epsilon_bar) at a given order.

    Parameters:
        epsilon (float): (Approximate) DP epsilon.
        alpha (float): Rényi order (> 1).
        delta (float): Target delta for (epsilon, delta)-DP.

    Returns:
        float: The corresponding Rényi DP budget (epsilon-bar).
    """
    return epsilon + math.log(delta) / (alpha - 1)


def privacy_budget_left(sampling_prob: float, steps: int, cur_sigma: float, delta: float, rdp_orders: int=32) -> tuple[float, float]:
    """
    Compute the consumed epsilon and its Rényi counterpart (epsilon-bar) for a Gaussian mechanism
    under Poisson subsampling, given current noise level and iterations.

    Parameters:
        sampling_prob (float): Sampling probability q per step (e.g., batch_size / dataset_size).
        steps (int | list[int]): Number of steps (iterations). If a list is supplied, its interpretation
            depends on the downstream accountant function usage.
        cur_sigma (float): Noise multiplier (std/sensitivity).
        delta (float): Target delta for (epsilon, delta)-DP.
        rdp_orders (int): Upper bound (exclusive) for even Rényi orders to evaluate, starting from 2.

    Returns:
        tuple[float, float]:
            - cur_eps (float): Spent (approximate) DP epsilon up to the specified steps.
            - epsilon_bar (float): Corresponding Rényi DP budget at the optimal order chosen by the accountant.
    """
    orders = np.arange(2, rdp_orders, 2.0)
    rdp = compute_rdp(sampling_prob, cur_sigma, steps, orders)
    cur_eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
    # print(f"Current eps: {cur_eps} optimal order {opt_order} current sigma {cur_sigma} rdp {rdp}")
    epsilon_bar = get_epsilon_bar_from_epsilon(cur_eps, opt_order, delta)
    # print(f"epsilon_bar: {epsilon_bar}")
    return float(cur_eps), epsilon_bar


def calc_privacy_spent_by_sigmas_and_probs(qlist: list[float], eps: float, delta: float, sigmas: Sequence[float], alpha: int=32) -> tuple[list[float], list[float]]:
    """
    Accumulate privacy spending across epochs with varying sampling probabilities and noise multipliers.

    Iteratively aggregates epsilon-bar over epochs, converts it to epsilon at a fixed alpha,
    and stops when the target epsilon would be exceeded.

    Parameters:
        qlist (list[float]): Sampling probabilities per epoch.
        eps (float): Target epsilon to not exceed.
        delta (float): Target delta for (epsilon, delta)-DP.
        sigmas (Sequence[float]): Noise multipliers (std/sensitivity) per epoch.
        alpha (float): Fixed Rényi order used to convert epsilon-bar to epsilon.

    Returns:
        tuple[list[float], list[float]]:
            - accumulated_epsilon_list: Epsilon after each epoch (monotone non-decreasing, truncated before exceeding eps).
            - accumulated_epsilon_bar_list: Corresponding accumulated epsilon-bar after each epoch.
    """
    assert len(qlist) == len(sigmas), (f'Expected qlist and sigmas to have the same length,'
                                       f' but got {len(qlist)} and {len(sigmas)} respectively.')
    accumulated_epsilon_bar, accumulated_epsilon = 0.0, 0.0
    accumulated_epsilon_bar_list, accumulated_epsilon_list = [], []
    steps_in_epoch = [int(1 / q) for q in qlist]
    pbar = tqdm(zip(sigmas, qlist), total=len(sigmas))
    for sigma, q in pbar:
        epsilon, epsilon_bar = privacy_budget_left(q, steps_in_epoch, sigma, delta)
        accumulated_epsilon_bar += epsilon_bar
        accumulated_epsilon = get_epsilon_from_epsilon_bar(accumulated_epsilon_bar, alpha, delta)
        if accumulated_epsilon > eps:
            break
        pbar.set_description(
            f"accumulated epsilon: {accumulated_epsilon} accumulated epsilon bar: {accumulated_epsilon_bar}")
        accumulated_epsilon_list.append(float(accumulated_epsilon))
        accumulated_epsilon_bar_list.append(float(accumulated_epsilon_bar))

    return accumulated_epsilon_list, accumulated_epsilon_bar_list


def calc_privacy_spent_by_sigma(q, eps, delta, sigmas, alpha=32):
    """
    Accumulate privacy spending across epochs with a fixed sampling probability and varying noise multipliers.

    Parameters:
        q (float): Sampling probability per step.
        eps (float): Target epsilon to not exceed.
        delta (float): Target delta for (epsilon, delta)-DP.
        sigmas (Sequence[float]): Noise multipliers (std/sensitivity) per epoch.
        alpha (float): Fixed Rényi order used to convert epsilon-bar to epsilon.

    Returns:
        tuple[list[float], list[float]]:
            - accumulated_epsilon_list: Epsilon after each epoch (monotone non-decreasing, truncated before exceeding eps).
            - accumulated_epsilon_bar_list: Corresponding accumulated epsilon-bar after each epoch.
    """
    accumulated_epsilon_bar, accumulated_epsilon = 0.0, 0.0
    accumulated_epsilon_bar_list, accumulated_epsilon_list = [], []
    steps_in_epoch = int(1 / q)
    pbar = trange(len(sigmas))
    for sigma in sigmas:
        epsilon, epsilon_bar = privacy_budget_left(q, steps_in_epoch, sigma, delta)
        accumulated_epsilon_bar += epsilon_bar
        accumulated_epsilon = get_epsilon_from_epsilon_bar(accumulated_epsilon_bar, alpha, delta)
        if accumulated_epsilon > eps:
            break
        pbar.set_description(
            f"accumulated epsilon: {accumulated_epsilon} accumulated epsilon bar: {accumulated_epsilon_bar}")
        accumulated_epsilon_list.append(float(accumulated_epsilon))
        accumulated_epsilon_bar_list.append(float(accumulated_epsilon_bar))

    return accumulated_epsilon_list, accumulated_epsilon_bar_list


def get_renyi_gaussian_sigma(sensitivity: float, alpha: float, epsilon_bar: float):
    """
    Compute the Gaussian noise multiplier (std/sensitivity) that satisfies a given Rényi DP budget.

    For the Gaussian mechanism under RDP, sigma = sqrt(alpha * sensitivity^2 / (2 * epsilon_bar)).

    Parameters:
        sensitivity (float): L2-sensitivity of the query/mechanism.
        alpha (float): Rényi order (> 1).
        epsilon_bar (float): Target Rényi DP budget at order alpha.

    Returns:
        float: Required noise multiplier (std/sensitivity).
    """
    return np.sqrt(np.array([((sensitivity ** 2.0) * alpha) / (2.0 * epsilon_bar)])).item()


def search_for_optimal_alpha(epsilon: float, deltas: list[float], alphas: list[float]):
    """
    For each delta, search over a set of Rényi orders to find the one inducing the smallest Gaussian sigma.

    Parameters:
        epsilon (float): Target (approximate) DP epsilon.
        deltas (list[float]): Candidate delta values.
        alphas (list[float]): Candidate Rényi orders to evaluate.

    Returns:
        tuple[dict[float, list[float]], dict[float, float]]:
            - sigmas_for_delta: Mapping delta -> list of sigma values corresponding to each alpha.
            - optimal_order_for_delta: Mapping delta -> alpha that minimizes sigma.
    """
    optimal_order_for_delta, sigmas_for_delta = {}, {}
    for delta in deltas:
        sigmas = [get_renyi_gaussian_sigma(sensitivity=1.0, alpha=alpha,
                                           epsilon_bar=get_epsilon_bar_from_epsilon(epsilon=epsilon, delta=delta,
                                                                                    alpha=alpha)) for alpha in alphas]
        # print(f"Sigmas for delta {delta}: {sigmas}")
        not_nan_sigmas = [sigma for sigma in sigmas if not np.isnan(sigma)]
        # print(f"Not nan sigmas for delta {delta}: {not_nan_sigmas}")
        min_val = min(not_nan_sigmas)
        # print(f"Min sigma value for delta {delta}: {min_val}")
        optimal_order = alphas[sigmas.index(min_val)]
        # print(f"Optimal order for delta {delta}: {optimal_order}")
        optimal_order_for_delta[delta] = optimal_order
        sigmas_for_delta[delta] = sigmas
    #     # optimal_orders_dict[8.0] = optimal_order
    #     if print_min_sigma:
    #         print(f'Delta {delta}: \t Minimal sigma value is {min_val} for order {optimal_order}. rdp (epsilon bar) is {get_epsilon_bar_from_epsilon(epsilon=epsilon, delta=delta, alpha=optimal_order)}')
    #     plt.plot(alphas, sigmas, label=f'delta {delta}');
    # plt.title(f'RDP values vs. order for epsilon={epsilon}');
    # plt.legend();
    # plt.xlabel("Reny'i Orders (alphas)");
    # plt.ylabel("Noise Multiplier (std/sensitivity)");
    return sigmas_for_delta, optimal_order_for_delta


def linear_decrease(upper_bound, lower_bound, num_values):
    """
    Generate a linearly decreasing sequence from upper_bound to lower_bound (exclusive of endpoint in step size).

    Parameters:
        upper_bound (float): Starting value.
        lower_bound (float): Ending value.
        num_values (int): Number of values to generate.

    Returns:
        list[float]: Sequence of length num_values decreasing approximately linearly.
    """
    diff = (upper_bound - lower_bound) / num_values
    return [upper_bound - diff * i for i in range(num_values)]


def geometric_decrease(upper_bound, lower_bound, num_values):
    """
    Generate a geometrically decreasing sequence from upper_bound to lower_bound.

    Parameters:
        upper_bound (float): Starting value (> 0).
        lower_bound (float): Ending value (> 0, < upper_bound).
        num_values (int): Number of values to generate.

    Returns:
        list[float]: Sequence of length num_values decreasing geometrically.
    """
    factor = (lower_bound / upper_bound) ** (1 / num_values)
    return [upper_bound * factor ** i for i in range(num_values)]


# def geometric_decrease(upper_bound, lower_bound, num_values, curvature_exponent=1.0):
#     base_factor = (lower_bound / upper_bound) ** (1 / num_values)
#     return [
#         upper_bound * (base_factor ** (i ** curvature_exponent))
#         for i in range(1, num_values+1)
#     ]
# def geometric_decrease(upper_bound, lower_bound, num_values, curvature_exponent=1.0):
#     return [
#         upper_bound * ((lower_bound / upper_bound) ** ((i / (num_values - 1)) ** curvature_exponent))
#         for i in range(num_values)
#     ]


def logarithmic_decrease(upper_bound, lower_bound, num_values):
    """
    Generate a decreasing sequence using a convex mapping of a normalized domain, yielding a
    faster drop early on and slower later (log-like behavior via convex functions).

    Parameters:
        upper_bound (float): Starting value.
        lower_bound (float): Ending value.
        num_values (int): Number of values to generate.

    Returns:
        list[float]: Sequence of length num_values decreasing with a convex profile.
    """
    # Convex functions
    funcs = {
        r"$x^2$": lambda x: x ** 2,
        r"$x^3$": lambda x: x ** 3,
        r"$e^x$ (normalized)": lambda x: (np.exp(x) - 1) / (np.e - 1)
    }

    def convex_subdivision(a, b, n, func):
        t = np.linspace(0, 1, n)
        return a - (a - b) * func(t)

    # return convex_subdivision(upper_bound, lower_bound, num_values, func=funcs[r"$x^2$"])
    return convex_subdivision(upper_bound, lower_bound, num_values, func=funcs[r"$x^3$"])
    # return convex_subdivision(upper_bound, lower_bound, num_values, func=funcs[r"$e^x$ (normalized)"])


def step_decrease(upper_bound, lower_bound, num_values, upper_ratio=0.84):
    """
    Generate a step-wise decreasing sequence with two plateaus.

    Parameters:
        upper_bound (float): Value for the initial plateau.
        lower_bound (float): Value for the final plateau.
        num_values (int): Total number of values to generate.
        upper_ratio (float): Fraction of the sequence at the upper plateau (in [0, 1]).

    Returns:
        list[float]: Sequence with an initial segment at upper_bound followed by lower_bound.
    """
    num_step_upper = int(upper_ratio * num_values)
    num_step_lower = num_values - num_step_upper
    sigam_factors = [upper_bound] * num_step_upper + [lower_bound] * num_step_lower
    return sigam_factors


def get_decrease_function(args):
    """
    Resolve a named decrease schedule function from CLI/args.

    Parameters:
        args: An object with attribute 'decrease_shape' being one of:
              {'linear', 'geometric', 'logarithmic', 'step'}.

    Returns:
        Callable: The corresponding schedule function taking (upper_bound, lower_bound, num_values).

    Raises:
        AssertionError: If args.decrease_shape is not a supported value.
    """
    get_decrease_function.hub = {'linear': linear_decrease,
                                 'geometric': geometric_decrease,
                                 'logarithmic': logarithmic_decrease,
                                 'step': step_decrease}
    assert args.decrease_shape in ['linear', 'geometric', 'logarithmic', 'step'], (
        f"Unknown decrease shape {args.decrease_shape}."
        f" Expected one of 'linear', 'geometric', 'logarithmic', 'step'.")
    return get_decrease_function.hub[args.decrease_shape]


def get_varying_sigma_values(q, n_epoch, eps, delta,
                             initial_sigma_factor, final_sigma_factor, decrease_func,
                             extra_noise_units=0, noise_for_step=0, alpha=32):
    """
    Construct a per-epoch sigma schedule that decreases over time, and compute cumulative privacy spending.

    The base sigma is computed to meet (eps, delta)-DP with fixed q over n_epoch if sigma were constant.
    The provided decrease_func scales this base sigma between initial_sigma_factor and final_sigma_factor.
    Optionally, additional noise can be added for a prefix of epochs.

    Parameters:
        q (float): Sampling probability per step.
        n_epoch (int): Number of epochs (length of the schedule).
        eps (float): Target epsilon for budgeting.
        delta (float): Target delta for (epsilon, delta)-DP.
        initial_sigma_factor (float): Multiplier for the initial epoch (> final_sigma_factor).
        final_sigma_factor (float): Multiplier for the final epoch (> 0).
        decrease_func (Callable): Function that returns a sequence of factors of length n_epoch.
        extra_noise_units (int, optional): Total units of additive noise to distribute at the beginning. Default: 0.
        noise_for_step (int, optional): Add this amount to each of the earliest epochs while budgeted by extra_noise_units. Default: 0.
        alpha (float, optional): Fixed Rényi order for converting epsilon-bar to epsilon. Default: 32.

    Returns:
        tuple[np.ndarray, list[float], list[float], float]:
            - sigmas (np.ndarray): The sigma schedule truncated so cumulative epsilon does not exceed eps.
            - accumulated_epsilon_list (list[float]): Epsilon after each kept epoch.
            - accumulated_epsilon_bar_list (list[float]): Epsilon-bar after each kept epoch.
            - sigma_orig (float): The base sigma before applying factors.

    Raises:
        AssertionError: If initial/final sigma factors are invalid.
    """
    assert initial_sigma_factor > final_sigma_factor, "Initial sigma factor must be greater than final sigma factor"
    assert final_sigma_factor > 0, "Final sigma factor must be greater than 0"

    steps_in_epoch = int(1 / q)
    sigma_orig, previous_eps = get_sigma(q=q, T=steps_in_epoch * n_epoch, eps=eps, delta=delta)
    decrease_factors = decrease_func(initial_sigma_factor, final_sigma_factor, n_epoch)
    sigmas = np.array([sigma_orig * sigma_factor for sigma_factor in decrease_factors])
    if extra_noise_units > 0:
        steps_to_add = extra_noise_units // noise_for_step
        sigmas[:steps_to_add] = sigmas[:steps_to_add] + noise_for_step
    accumulated_epsilon_list, accumulated_epsilon_bar_list = calc_privacy_spent_by_sigma(q, eps, delta, sigmas, alpha)
    num_epochs_to_reach_eps = len(accumulated_epsilon_list)
    return sigmas[:num_epochs_to_reach_eps], accumulated_epsilon_list, accumulated_epsilon_bar_list, sigma_orig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    batchsize = 1024
    n_training = 50_000
    n_epoch = 200
    delta = 1 / n_training
    epsilon = 8
    # initial_sigma_factor = 1.0
    # final_sigma_factor = 0.3
    # # final_sigma_factor = 0.64
    sampling_prob = batchsize / n_training
    first_sampling_prob = sampling_prob
    steps = int(n_epoch / sampling_prob)
    # alphas = list(range(2, 100))
    alpha = 32   # 32

    cur_sigma, previous_eps = get_sigma(q=sampling_prob, T=steps, eps=epsilon, delta=delta)
    print(f"cur sigma: {cur_sigma}")
    print(f"previous eps: {previous_eps}")

    base_sigma = 1.53
    switch_epoch = 128
    sigmas = [base_sigma] * switch_epoch
    accumulated_epsilon_list, accumulated_epsilon_bar_list = calc_privacy_spent_by_sigma(q = sampling_prob,
                                                                                         eps = epsilon,
                                                                                         delta = delta,
                                                                                         sigmas = sigmas,
                                                                                         alpha = alpha)
    print(f"Accumulated epsilon list: {accumulated_epsilon_list}")
    print(f"Accumulated epsilon bar list: {accumulated_epsilon_bar_list}")
    print(f"Final epsilon: {accumulated_epsilon_list[-1]}")
    print(f"Final epsilon bar: {accumulated_epsilon_bar_list[-1]}")
    total_renyi_budget = get_epsilon_bar_from_epsilon(epsilon=epsilon, alpha=alpha, delta=delta)
    print(f"Total RDP budget: {total_renyi_budget}")
    left_renyi_budget = total_renyi_budget - accumulated_epsilon_bar_list[-1]
    print(f"Left RDP budget: {left_renyi_budget}")
    left_eps = get_epsilon_from_epsilon_bar(left_renyi_budget, alpha=alpha, delta=delta)

    batchsize = 256
    n_training = 50_000
    n_epoch = 200 - 128
    delta = 1 / n_training
    # epsilon = left_eps
    initial_sigma_factor = 1.0
    final_sigma_factor = 0.3
    # final_sigma_factor = 0.64
    sampling_prob = batchsize / n_training
    second_sampling_prob = sampling_prob
    steps = int(n_epoch / sampling_prob)
    alphas = list(range(2, 100))

    sigmas, accumulated_epsilon, accumulated_epsilon_bar, sigma_orig = get_varying_sigma_values(sampling_prob,
                                                                                                int(n_epoch),
                                                                                                eps = epsilon,
                                                                                                delta = delta,
                                                                                                initial_sigma_factor=initial_sigma_factor,
                                                                                                final_sigma_factor=final_sigma_factor,
                                                                                                decrease_func=linear_decrease,
                                                                                                alpha=alpha)
    print(f"accumulated epsilon bar: {accumulated_epsilon_bar}")
    print(f'accumulated epsilon: {accumulated_epsilon}')
    print(f'sigmas: {sigmas}')
    print(f"Number of sigmas: {len(sigmas)}")
    sigmas_above_orig = np.array(sigmas) > sigma_orig
    num_sigmas_above_orig = sum(sigmas_above_orig)
    print(f"Number of sigmas above original sigma: {num_sigmas_above_orig}")
    sigmas = [base_sigma] * switch_epoch + sigmas.tolist()
    qlist = [first_sampling_prob] * switch_epoch + [second_sampling_prob] * (len(sigmas) - switch_epoch)
    print(f"Number of sigmas: {len(sigmas)}")

    accumulated_epsilon_list, accumulated_epsilon_bar_list = calc_privacy_spent_by_sigma(q=sampling_prob,
                                                                                         eps=epsilon,
                                                                                         delta=delta,
                                                                                         sigmas=sigmas,
                                                                                         alpha=alpha)


    print(f"Accumulated epsilon list: {accumulated_epsilon_list}")
    print(f"Accumulated epsilon bar list: {accumulated_epsilon_bar_list}")
    print(f"Final epsilon: {accumulated_epsilon_list[-1]}")
    print(f"Final epsilon bar: {accumulated_epsilon_bar_list[-1]}")

    raise Exception("Stop")


    # sigmas, optimal_orders = search_for_optimal_alpha(epsilon=epsilon, deltas=[delta], alphas=alphas)
    # not_nan_sigmas = [sigma.item() for sigma in sigmas if not np.isnan(sigma)]
    # print(f"Not nan sigmas for delta {delta}: {not_nan_sigmas}")
    # min_val = min(not_nan_sigmas)
    # print(f"Min sigma value for delta {delta}: {min_v
    # optimal_order = alphas[sigmas.index(min_val)]
    # np_alphas = np.array(alphas)
    # optimal_order_index = np.argwhere(np_alphas == optimal_orders[delta]).item()
    # print(f'optimal ordr index: {optimal_order_index}')
    # sigma_optimal = sigmas[delta][optimal_order_index]
    # print(f"Optimal sigma for delta {delta}: {sigma_optimal}")
    # thirty_two_order = alphas[sigmas[delta].index(32.0)]
    # alphas = [32, 23, 23.1, 23.13, 23.129, 23.1285]
    # for alpha in alphas:
    #     thirty_two_order_index = np.argwhere(np_alphas == alpha)
    #     assert thirty_two_order_index.size <= 1, f"There is more than one index for alpha {alpha}"
    #     thirty_two_order_index = thirty_two_order_index.item() if thirty_two_order_index.size > 0 else None
    #     print(f'{alpha}_order index: {thirty_two_order_index}')
    #
    #     sigma_suboptimal = sigmas[delta][thirty_two_order_index] if thirty_two_order_index is not None else None
    #
    #     print(f"{alpha} sigma for delta {delta}: {sigma_suboptimal}")
    #
    # #
    # print(f"Optimal orders for (epsilon, delta) ({epsilon}, {delta}): {optimal_orders}")
    #
    # #
    # raise Exception("Stop")

    # Plot
    plt.figure(figsize=(10, 6))

    # curvatures = [0.5, 1.0]
    # curvatures = [1.0, 0.8,  1.1]
    # geometric_decrease_funcs = [partial(geometric_decrease, curvature_exponent=v) for v in curvatures]
    # for decrease_function in [linear_decrease, geometric_decrease, logarithmic_decrease]:
    # for decrease_function in [linear_decrease]:
    for decrease_function in [step_decrease]:
        # for crv, decrease_function in zip(curvatures, geometric_decrease_funcs):
        #     for extra_noise_units in [0, 10000]:
        #         if extra_noise_units == 0:
        #             continue
        #         for noise_for_step in [100, 1000]:
        for alpha in [23, 32]:
            # for decrease_function in [concave_decrease]:
            sigmas, accumulated_epsilon, accumulated_epsilon_bar, sigma_orig = get_varying_sigma_values(sampling_prob,
                                                                                                        int(n_epoch),
                                                                                                        epsilon,
                                                                                                        delta,
                                                                                                        initial_sigma_factor=initial_sigma_factor,
                                                                                                        final_sigma_factor=final_sigma_factor,
                                                                                                        decrease_func=decrease_function,
                                                                                                        alpha=alpha
                                                                                                        )
            # extra_noise_units=extra_noise_units,
            # noise_for_step=noise_for_step)
            # print(f"Decrease Function geometric curvature {crv} extra noise units {extra_noise_units}, {noise_for_step} for step")
            print(f"Decrease Function {decrease_function.__name__} alpha {alpha}")
            print('**************************************************')
            print(f"Number of sigmas: {len(sigmas)}")
            print(f'First sigma: {sigmas[0]}')
            print(f"Final sigma: {sigmas[-1]}")
            print(f'original sigma: {sigma_orig}')
            sigmas_above_orig = np.array(sigmas) > sigma_orig
            print(f"Number of sigmas above original sigma: {sum(sigmas_above_orig)}")
            print(f"Accumulated epsilons: {accumulated_epsilon}")
            print(f"Accumulated epsilon-bars: {accumulated_epsilon_bar}")
            print(f'Accumulated epsilon bar: {accumulated_epsilon_bar[-1]}')
            print(f'Accumulated epsilon    : {accumulated_epsilon[-1]}')

            # plt.plot(range(len(sigmas)), sigmas, label=f'geometric curvature exponent {crv}')
            # plt.scatter(range(len(sigmas)), sigmas, label=f'geometric curv exp {crv} extra {extra_noise_units}, {noise_for_step} for step')
            plt.plot(range(len(sigmas)), sigmas, label=f'{decrease_function.__name__}_alpha_{alpha}')

    plt.title(f"Sigma factor decrease from {initial_sigma_factor} to {final_sigma_factor}")
    plt.xlabel("Subdivision index")
    plt.ylabel("Value")
    plt.xlim(0, 201)
    plt.ylim(0, 10)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
