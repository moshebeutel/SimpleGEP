import math
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


def calc_privacy_spent_by_sigma(q, eps, delta, sigmas):
    accumulated_epsilon_bar, accumulated_epsilon = 0.0, 0.0
    accumulated_epsilon_bar_list, accumulated_epsilon_list = [], []
    steps_in_epoch = int(1 / q)
    pbar = trange(len(sigmas))
    for sigma in sigmas:
        epsilon, epsilon_bar = privacy_budget_left(q, steps_in_epoch, sigma, delta)
        accumulated_epsilon_bar += epsilon_bar
        accumulated_epsilon = get_epsilon_from_epsilon_bar(accumulated_epsilon_bar, 32, delta)
        pbar.set_description(
            f"accumulated epsilon: {accumulated_epsilon} accumulated epsilon bar: {accumulated_epsilon_bar}")
        if accumulated_epsilon > eps:
            break
        accumulated_epsilon_list.append(float(accumulated_epsilon))
        accumulated_epsilon_bar_list.append(float(accumulated_epsilon_bar))

    return accumulated_epsilon_list, accumulated_epsilon_bar_list


def linear_decrease(upper_bound, lower_bound, num_values):
    diff = (upper_bound - lower_bound) / num_values
    return [upper_bound - diff * i for i in range(num_values)]


def geometric_decrease(upper_bound, lower_bound, num_values):
    factor = (lower_bound / upper_bound) ** (1 / num_values)
    return [upper_bound * factor ** i for i in range(num_values)]


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


def get_decrease_function(args):
    get_decrease_function.hub = {'linear': linear_decrease, 'geometric': geometric_decrease,
                                 'logarithmic': logarithmic_decrease}
    assert args.decrease_shape in ['linear', 'geometric', 'logarithmic'], (
        f"Unknown decrease shape {args.decrease_shape}."
        f" Expected one of 'linear', 'geometric', 'logarithmic'.")
    return get_decrease_function.hub[args.decrease_shape]


def get_varying_sigma_values(q, n_epoch, eps, delta, initial_sigma_factor, final_sigma_factor, decrease_func):
    assert initial_sigma_factor > final_sigma_factor, "Initial sigma factor must be greater than final sigma factor"
    assert final_sigma_factor > 0, "Final sigma factor must be greater than 0"

    steps_in_epoch = int(1 / q)
    sigma_orig, previous_eps = get_sigma(q=q, T=steps_in_epoch * n_epoch, eps=eps, delta=delta)
    decrease_factors = decrease_func(initial_sigma_factor, final_sigma_factor, n_epoch)
    sigmas = [sigma_orig * sigma_factor for sigma_factor in decrease_factors]
    accumulated_epsilon_list, accumulated_epsilon_bar_list = calc_privacy_spent_by_sigma(q, eps, delta, sigmas)
    num_epochs_to_reach_eps = len(accumulated_epsilon_list)
    return sigmas[:num_epochs_to_reach_eps], accumulated_epsilon_list, accumulated_epsilon_bar_list, sigma_orig


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    batchsize = 256
    n_training = 50000
    n_epoch = 25
    delta = 1 / n_epoch
    epsilon = 1
    initial_sigma_factor = 3.2
    final_sigma_factor = 0.4
    sampling_prob = batchsize / n_training
    steps = int(n_epoch / sampling_prob)

    # Plot
    plt.figure(figsize=(10, 6))

    for decrease_function in [linear_decrease, geometric_decrease, logarithmic_decrease]:
        # for decrease_function in [concave_decrease]:
        sigmas, accumulated_epsilon, accumulated_epsilon_bar, sigma_orig = get_varying_sigma_values(sampling_prob,
                                                                                                    n_epoch, epsilon,
                                                                                                    delta,
                                                                                                    initial_sigma_factor=initial_sigma_factor,
                                                                                                    final_sigma_factor=final_sigma_factor,
                                                                                                    decrease_func=decrease_function)
        print(f"Decrease Function {decrease_function.__name__}")
        print('**************************************************')
        print(f"Number of sigmas: {len(sigmas)}")
        print(f'First sigma: {sigmas[0]}')
        print(f"Final sigma: {sigmas[-1]}")
        print(f'original sigma: {sigma_orig}')
        sigmas_above_orig = np.array(sigmas) > sigma_orig
        print(f"Number of sigmas above original sigma: {sum(sigmas_above_orig)}")
        print(f"Accumulated epsilons: {accumulated_epsilon}")
        print(f"Accumulated epsilon-bars: {accumulated_epsilon_bar}")

        plt.plot(range(len(sigmas)), sigmas, label=decrease_function.__name__)

    plt.title(f"Sigma factor decrease from {initial_sigma_factor} to {final_sigma_factor}")
    plt.xlabel("Subdivision index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
