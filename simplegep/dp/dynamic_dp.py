import math
from functools import partial

import numpy as np
from tqdm import trange
from simplegep.dp.rdp_accountant import compute_rdp, get_privacy_spent, get_sigma


def get_epsilon_from_epsilon_bar(epsilon_bar: float, alpha: float, delta: float):
    return epsilon_bar - math.log(delta) / (alpha - 1)


def get_epsilon_bar_from_epsilon(epsilon: float, alpha: float, delta: float):
    return epsilon + math.log(delta) / (alpha - 1)


def privacy_budget_left(sampling_prob, steps, cur_sigma, delta, rdp_orders=32):
    orders = np.arange(2, rdp_orders, 2.0)
    rdp = compute_rdp(sampling_prob, cur_sigma, steps, orders)
    cur_eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
    # print(f"Current eps: {cur_eps} optimal order {opt_order} current sigma {cur_sigma} rdp {rdp}")
    epsilon_bar = get_epsilon_bar_from_epsilon(cur_eps, opt_order, delta)
    # print(f"epsilon_bar: {epsilon_bar}")
    return float(cur_eps), epsilon_bar


def calc_privacy_spent_by_sigma(q, eps, delta, sigmas, alpha=32):
    accumulated_epsilon_bar, accumulated_epsilon = 0.0, 0.0
    accumulated_epsilon_bar_list, accumulated_epsilon_list = [], []
    steps_in_epoch = int(1 / q)
    pbar = trange(len(sigmas))
    for sigma in sigmas:
        epsilon, epsilon_bar = privacy_budget_left(q, steps_in_epoch, sigma, delta)
        accumulated_epsilon_bar += epsilon_bar
        accumulated_epsilon = get_epsilon_from_epsilon_bar(accumulated_epsilon_bar, alpha, delta)
        pbar.set_description(
            f"accumulated epsilon: {accumulated_epsilon} accumulated epsilon bar: {accumulated_epsilon_bar}")
        if accumulated_epsilon > eps:
            break
        accumulated_epsilon_list.append(float(accumulated_epsilon))
        accumulated_epsilon_bar_list.append(float(accumulated_epsilon_bar))

    return accumulated_epsilon_list, accumulated_epsilon_bar_list


def get_renyi_gaussian_sigma(sensitivity: float, alpha: float, epsilon_bar: float):
    return np.sqrt(np.array([((sensitivity ** 2.0) * alpha) / (2.0 * epsilon_bar)])).item()


def search_for_optimal_alpha(epsilon: float, deltas: list[float], alphas: list[float]):
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
    diff = (upper_bound - lower_bound) / num_values
    return [upper_bound - diff * i for i in range(num_values)]


def geometric_decrease(upper_bound, lower_bound, num_values):
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
    num_step_upper = int(upper_ratio * num_values)
    num_step_lower = num_values - num_step_upper
    sigam_factors = [upper_bound] * num_step_upper + [lower_bound] * num_step_lower
    return sigam_factors


def get_decrease_function(args):
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

    batchsize = 32
    n_training = 50_000
    n_epoch = 200
    delta = 1 / n_training
    epsilon = 8
    initial_sigma_factor = 3.2
    final_sigma_factor = 0.61
    sampling_prob = batchsize / n_training
    steps = int(n_epoch / sampling_prob)
    alphas = list(range(2, 100))
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
