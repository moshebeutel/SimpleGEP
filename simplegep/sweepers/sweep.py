from functools import partial
from logging import Logger
from pathlib import Path
from typing import Dict
import wandb
import yaml

from simplegep.utils import set_seed, parse_args, set_logger

def load_config(file_path: str)-> Dict:
    """
    Loads the configuration from a specified YAML file.

    This function checks the existence, type, and extension of the provided
    file path to ensure it is a valid YAML configuration file. Once validated,
    it reads and parses the YAML content, returning the configuration data.

    Args:
        file_path (str): The path to the YAML configuration file.

    Returns:
        dict: The parsed configuration data as a dictionary.

    Raises:
        AssertionError: If the file does not exist, is not a file, or does not
            have the '.yaml' extension.
    """
    config_file_path = Path(file_path)
    assert config_file_path.exists(), f"config file {config_file_path} does not exist"
    assert config_file_path.is_file(), f"config file {config_file_path} is not a file"
    assert config_file_path.suffix == ".yaml", f"config file {config_file_path} is not a yaml file"
    with open(config_file_path, 'r') as stream:
        sweep_config = yaml.safe_load(stream)
    return sweep_config

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
    logger = init_logger(args)

    default_parameters = {
            "lr": {"values": [1e-3]},
            "seed": {"values": [3]},
            "clip_value": {"values": [35.0]},
            "clip_strategy": {"values": ["value"]},
            "eps": {"values": [args.eps]},
            "optimizer": {"values": ["adam"]},
            "momentum": {"values": [0.9]},
            "weight_decay": {"values": [1e-4]},
            "filters": {"values": [4]},
            "embedder": {"values": ["svd"]},
            "dynamic_noise": {"values": [False]},
            "dynamic_noise_high_factor": {"values": [3.2]},
            "dynamic_noise_low_factor": {"values": [0.3]},
            "decrease_shape": {"values": ["linear"]},
            "num_epochs": {"values": [1000]},
            # "stop_embedding_epoch": {"values": [100]},
            "num_basis": {"values": [2000]},
            "grads_history_size": {"values": [0]},
            "aux_data_size": {"values": [2000]},
            "batchsize": {"values": [512]}
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
        "num_basis": {"values": [10, 200]},
        "embedder": {"values": ["svd", "kernel_pca"]},
        "grads_history_size": {"values": [1000, 3000]},
    }

    gep_kernel_pca_parameters = {
        "embedder": {"values": ["kernel_pca"]},
        "kernel_type": {"values": ["linear", "rbf", "poly", "cosine", "sigmoid"]},
    }



    sweep_configuration = {
        "name": f"{args.dp_method.upper()}_SEED_{args.seed}_{args.model_name.upper()}_EPS_{args.eps}",
        "method": "grid",
        "metric": {"goal": "maximize", "name": "test_acc"},
        "parameters": {
            **default_parameters,
            # **optimizer_parameters,
            # **dp_parameters,
            # **gep_parameters,
            # **gep_kernel_pca_parameters,
            # **dynamic_noise_parameters
        }
    }

    sweep_configuration = {
        "name": f"{args.dp_method.upper()}_SEED_{args.seed}_{args.dataset.upper()}_EPS_{args.eps}",
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "test_acc"},
        "early_terminate": {"eta": 3, "s": 2, "type": "hyperband", "min_iter": 5},
        "parameters": {
            "lr": {"min": 1e-2, "max": 0.1},
            "seed": {"values": [args.seed]},
            "clip_value": {"min": 1.0e-4, "max": 1.0},
            "num_basis": {"min": 50, "max": 1000},
            "num_epochs": {"min": 10, "max": 100},
            "clip_strategy": {"values": ["value"]},
            "batchsize": {"values": [1024, 512]},
        }
    }
    # wandb.login()

    sweep(sweep_config=sweep_configuration, args=args,
          train_fn=partial(train, logger=logger))


def init_logger(args) -> Logger:
    logger = set_logger(logger_name=args.sess, log_dir='log', level='DEBUG')
    logger.info(f'Logger is set - session: {args.sess}')
    logger.info(f'Arguments: {args}')
    return logger


def prepare_sweep(data_name, dp_method, config_yaml_path):
    args = parse_args(data_name=data_name, dp_method=dp_method)
    logger = init_logger(args)
    working_dir = Path(__file__).resolve().parents[2]
    logger.debug(f'Working dir {working_dir}')
    config_path =  working_dir / config_yaml_path
    assert config_path.exists(), f'config file {config_path} does not exist'
    seed_parameters = {
        "seed": {"values": [args.seed]}}
    sweep_configuration = load_config(config_path.as_posix())
    sweep_configuration['parameters'].update(seed_parameters)
    sweep_name = f"eps{args.eps}_epochs{args.num_epochs}_{dp_method.upper()}_{args.dataset.upper()}_seed{args.seed}"
    if args.use_gp:
        sweep_name = f"GP_{sweep_name}"
    sweep_configuration['name'] = sweep_name
    return sweep_configuration, args, logger


if __name__ == '__main__':
    working_dir = Path(__file__).resolve().parents[2]
    args = parse_args(description=f'Differentially Private learning sweep')

    if args.dp_method == 'gep':
        from simplegep.trainers.gep_trainer import train
    elif args.dp_method == 'dp_sgd':
        from simplegep.trainers.dp_sgd_trainer import train
    elif args.dp_method == 'super':
        from simplegep.trainers.super_trainer import train

        args.dynamic_noise = True
        args.decrease_shape = 'step'
    else:
        assert args.dp_method == 'no_dp', f'dp_method {args.dp_method} unknown'
        from simplegep.trainers.no_dp_trainer import train

    main(args)
