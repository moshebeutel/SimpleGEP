import time
from functools import partial
import wandb
from simplegep.utils import set_seed, parse_args, set_logger


def sweep_train(sweep_id, args, train_fn, config=None):
    with wandb.init(config=config):
        config = wandb.config
        config.update({'sweep_id': sweep_id})
        set_seed(config.seed)

        for k, v in config.items():
            if k in args:
                setattr(args, k, v)

        wandb.run.name = '_'.join([f'{k}_{v}' for k, v in config.items()])
        train_fn(args)


def init_sweep(config):
    sweep_id = wandb.sweep(sweep=config, project="GEP")
    return sweep_id


def start_sweep(sweep_id, f_sweep):
    wandb.agent(sweep_id=sweep_id, function=f_sweep)


def sweep(sweep_config, args, train_fn):
    sweep_id = init_sweep(sweep_config)
    f_sweep = partial(sweep_train, sweep_id=sweep_id, args=args, train_fn=train_fn)
    # wandb.agent(sweep_id=sweep_id, function=f_sweep)
    start_sweep(sweep_id, f_sweep)


def main(args):
    logger = set_logger(logger_name=args.sess, log_dir='log', level='DEBUG')
    logger.info(f'Logger is set - session: {args.sess}')
    logger.info(f'Arguments: {args}')

    default_parameters = {
            "lr": {"values": [1e-4]},
            "seed": {"values": [3]},
            "clip_value": {"values": [35.0]},
            "clip_strategy": {"values": ["median"]},
            "eps": {"values": [args.eps]},
            "optimizer": {"values": ["adam"]},
            "momentum": {"values": [0.9]},
            "filters": {"values": [4]},
            "embedder": {"values": ["svd"]},
            "dynamic_noise": {"values": [True] },
            "dynamic_noise_high_factor": {"values": [3.2]},
            "dynamic_noise_low_factor": {"values": [0.4]},
            "decrease_shape": {"values": ["geometric"]},
            "num_epochs": {"values": [25]},
            "num_bases": {"values": [200]},
            "aux_data_size": {"values": [2000]},
            "batchsize": {"values": [256]}
    }

    dynamic_noise_parameters = {
        "dynamic_noise": {"values": [True]},
        "dynamic_noise_high_factor": {"values": [1.1, 1.2]},
        "dynamic_noise_low_factor": {"values": [0.6]},
        "decrease_shape": {"values": ["linear", "geometric", "logarithmic"]}
    }

    optimizer_parameters = {
        "optimizer": {"values": ["adam", "sgd"]},
        "momentum": {"values": [0.9]},
        "weight_decay": {"values": [0.0001, 0.001, 0.01]},
        "lr": {"values": [1e-4, 5e-5]},
    }

    dp_parameters = {
        "eps": {"values": [args.eps]},
        "clip_value": {"values": [35.0, 5.0]},
        "clip_strategy": {"values": ["value", "median", "max"]},
    }

    gep_parameters = {
        "num_bases": {"values": [100, 200]},
        "embedder": {"values": ["svd", "kernel_pca"]},
    }



    sweep_configuration = {
        "name": f"{args.dp_method.upper()}_SEED_{args.seed}_TINY_EPS_{args.eps}",
        "method": "grid",
        "metric": {"goal": "maximize", "name": "test_acc"},
        "parameters": {
            **default_parameters,
            **optimizer_parameters,
            # **dp_parameters,
            # **gep_parameters,
            # **dynamic_noise_parameters
        }
    }

    # wandb.login()

    sweep(sweep_config=sweep_configuration, args=args,
          train_fn=partial(train, logger=logger))


if __name__ == '__main__':
    args = parse_args(description=f'Differentially Private learning sweep')

    if args.dp_method == 'gep':
        from simplegep.trainers.gep_trainer import train
    else:
        assert args.dp_method == 'dp_sgd'
        from simplegep.trainers.dp_sgd_trainer import train
    main(args)
