import logging
from copy import copy
import numpy as np
from simplegep.data.cifar_loader import get_num_samples
from simplegep.dp.dp_params import get_dp_params
from simplegep.dp.dynamic_dp import get_decrease_function, get_varying_sigma_values, get_epsilon_from_epsilon_bar
from simplegep.trainers.dp_sgd_trainer import train as dp_sgd_train
from simplegep.trainers.gep_trainer import train as gep_train


def analyze_step(values: list):
    np_vals = np.array(values)
    uniq = np.unique_counts(np_vals)
    assert len(uniq.values) == 2
    switch_index = uniq.counts[1]
    assert values[switch_index - 1] == values[0]
    assert values[switch_index] == values[-1]
    return switch_index, uniq.values, uniq.counts


def train(args, logger: logging.Logger):
    logger.info('Super trainer start')

    num_epochs = args.num_epochs
    logger.debug(f'num epochs {num_epochs}')
    dp_params = get_dp_params(batchsize=args.batchsize,
                              num_training_samples=get_num_samples(),
                              num_epochs=args.num_epochs,
                              epsilon=args.eps)
    logger.debug(f'Super trainer DP params set: {dp_params}')
    assert args.dynamic_noise, f'Expected dynamic noise to be set for super trainer. Got {args.dynamic_noise} != True'
    sigma_decrease_function = get_decrease_function(args)
    logger.debug(f'Using decrease function {sigma_decrease_function.__name__}')
    sigma_list, accumulated_epsilon_list, accumulated_epsilon_bar_list, sigma_orig = (
        get_varying_sigma_values(q=dp_params.sampling_prob,
                                 n_epoch=args.num_epochs,
                                 eps=args.eps, delta=dp_params.delta,
                                 initial_sigma_factor=args.dynamic_noise_high_factor,
                                 final_sigma_factor=args.dynamic_noise_low_factor,
                                 decrease_func=sigma_decrease_function, ))
    logger.debug(f'Created varying sigma list with {len(sigma_list)} values')
    switch_trains_epoch, values, counts = analyze_step(sigma_list)
    logger.debug(f'switch trains epoch {switch_trains_epoch} values {values} counts {counts}')
    assert len(sigma_list) > switch_trains_epoch, f'Expected sigma list to have more sigmas than switch trains epoch. '
    assert switch_trains_epoch < num_epochs, f'Expected switch train epoch before num_epochs.' \
                                             f' Got switch epoch {switch_trains_epoch} num epochs {num_epochs}'

    gep_epsilon = accumulated_epsilon_list[switch_trains_epoch - 1]
    dp_sgd_epsilon_bar = accumulated_epsilon_bar_list[-1] - accumulated_epsilon_bar_list[switch_trains_epoch - 1]
    dp_sgd_epsilon = get_epsilon_from_epsilon_bar(epsilon_bar=dp_sgd_epsilon_bar, alpha=32, delta=dp_params.delta)
    logger.info(f'Created varying sigma list with {len(sigma_list)} values')
    logger.debug(f'Sigma list: {sigma_list}')
    logger.debug(f'Accumulated epsilon list: {accumulated_epsilon_list}')
    logger.debug(f'Accumulated epsilon bar list: {accumulated_epsilon_bar_list}')
    logger.debug(f'Sigma orig: {sigma_orig}')

    dp_sgd_args = copy(args)
    gep_args = copy(args)
    dp_sgd_args.dynamic_noise = False
    gep_args.dynamic_noise = False

    gep_args.num_epochs = switch_trains_epoch
    gep_args.eps = gep_epsilon
    dp_sgd_args.eps = dp_sgd_epsilon
    logger.info(f'gep epsilon {gep_epsilon} dp sgd epsilon {dp_sgd_epsilon}')
    logger.info('Call GEP train')
    gep_acc, checkpoint_name = gep_train(args=gep_args, logger=logger)
    logger.info(f'gep train part ended with acc {gep_acc}')

    dp_sgd_args.resume = True
    dp_sgd_args.checkpoint = checkpoint_name
    dp_sgd_args.optimizer = 'adam'
    dp_sgd_args.lr = 1e-3
    logger.info('Call DP_SGD train')
    dp_sgd_acc, checkpoint_name = dp_sgd_train(args=dp_sgd_args, logger=logger)
    logger.info(f'dp sgd train ended with acc {dp_sgd_acc}')
    logger.info('super train ended')
