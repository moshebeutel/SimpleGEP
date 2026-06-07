import os
from functools import partial
import wandb

from simplegep.sweepers.sweep import prepare_sweep, sweep
from simplegep.utils import parse_args, set_logger, set_seed





def init_logger(args):
    logger = set_logger(logger_name=args.sess, log_dir=args.log_root, level=args.log_level)
    logger.info(f'Logger is set - session: {args.sess}')
    logger.info(f'Arguments: {args}')
    return logger

def start_train(args, train_fn):
    logger = init_logger(args)
    set_seed(args.seed)
    with wandb.init(project='GEP', name=args.sess):
        wandb.config.update(vars(args))
        run_name_fields = ['dp_method', 'dataset', 'eps', 'lr', 'batchsize', 'dynamic_noise']
        if args.dp_method == 'gep':
            run_name_fields.extend(['num_basis', 'grads_history_size'])
        wandb.run.name = '_'.join([f'{k}_{getattr(args, k)}'.upper() for k in run_name_fields])
        try:
            train_fn(args, logger)
        except Exception as e:
            logger.error(f'Error in training during sweep: {e}')



# CIFAR10 runners
def run_no_dp_cifar10():
    args = parse_args(data_name='cifar10', dp_method='no_dp')
    from simplegep.trainers.no_dp_trainer import train
    start_train(args, train_fn=train)


def run_dp_sgd_cifar10():
    args = parse_args(data_name='cifar10', dp_method='dp_sgd')
    from simplegep.trainers.dp_sgd_trainer import train
    start_train(args, train_fn=train)

def run_gp_dp_sgd_cifar10():
    os.environ['USE_GP'] = 'true'
    args = parse_args(data_name='cifar10', dp_method='dp_sgd')
    from simplegep.gp_trainers.gp_dp_sgd_trainer import train
    start_train(args, train_fn=train)

def run_gep_cifar10():
    args = parse_args(data_name='cifar10', dp_method='gep')
    from simplegep.trainers.gep_trainer import train
    start_train(args, train_fn=train)

def sweep_no_dp_cifar10():
    config_yaml_path = 'simplegep/sweepers/sweep_configurations/sgd_dp_bayes.yaml'
    sweep_configuration, args, logger = prepare_sweep(data_name='cifar10', dp_method='no_dp',
                                                      config_yaml_path=config_yaml_path)
    from simplegep.trainers.no_dp_trainer import train

    sweep(sweep_config=sweep_configuration, args=args,
          train_fn=partial(train, logger=logger))

def sweep_dp_sgd_cifar10():
    config_yaml_path = 'simplegep/sweepers/sweep_configurations/sgd_dp_bayes.yaml'
    sweep_configuration, args, logger = prepare_sweep(data_name='cifar10', dp_method='dp_sgd',
                                                      config_yaml_path=config_yaml_path)
    from simplegep.trainers.dp_sgd_trainer import train

    sweep(sweep_config=sweep_configuration, args=args,
          train_fn=partial(train, logger=logger))


def sweep_gep_cifar10():
    config_yaml_path = 'simplegep/sweepers/sweep_configurations/gep_bayes.yaml'
    sweep_configuration, args, logger = prepare_sweep(data_name='cifar10', dp_method='gep',
                                                      config_yaml_path=config_yaml_path)
    from simplegep.trainers.gep_trainer import train

    sweep(sweep_config=sweep_configuration, args=args,
          train_fn=partial(train, logger=logger))


# putEMG runners
def run_no_dp_putemg():
    args = parse_args(data_name='putemg', dp_method='no_dp')
    from simplegep.trainers.no_dp_trainer import train
    start_train(args, train_fn=train)

def run_dp_sgd_putemg():
    args = parse_args(data_name='putemg', dp_method='dp_sgd')
    from simplegep.trainers.dp_sgd_trainer import train
    start_train(args, train_fn=train)


def run_gp_dp_sgd_putemg():
    os.environ['USE_GP'] = 'true'
    args = parse_args(data_name='putemg', dp_method='dp_sgd')
    from simplegep.gp_trainers.gp_dp_sgd_trainer import train
    start_train(args, train_fn=train)

def sweep_no_dp_putemg():
    config_yaml_path = 'simplegep/sweepers/sweep_configurations/sgd_dp_bayes.yaml'
    sweep_configuration, args, logger = prepare_sweep(data_name='putemg', dp_method='no_dp',
                                                      config_yaml_path=config_yaml_path)
    from simplegep.trainers.no_dp_trainer import train

    sweep(sweep_config=sweep_configuration, args=args,
          train_fn=partial(train, logger=logger))

def sweep_dp_sgd_putemg():
    config_yaml_path = 'simplegep/sweepers/sweep_configurations/sgd_dp_bayes.yaml'
    sweep_configuration, args, logger = prepare_sweep(data_name='putemg', dp_method='dp_sgd',
                                                      config_yaml_path=config_yaml_path)
    from simplegep.trainers.dp_sgd_trainer import train

    sweep(sweep_config=sweep_configuration, args=args,
          train_fn=partial(train, logger=logger))

def sweep_gep_putemg():
    config_yaml_path = 'simplegep/sweepers/sweep_configurations/gep_bayes.yaml'
    sweep_configuration, args, logger = prepare_sweep(data_name='putemg', dp_method='gep',
                                                      config_yaml_path=config_yaml_path)
    from simplegep.trainers.gep_trainer import train

    sweep(sweep_config=sweep_configuration, args=args,
          train_fn=partial(train, logger=logger))

# UHN Typing Dataset (keypressemg) runners
def run_no_dp_keypressemg():
    args = parse_args(data_name='keypressemg', dp_method='no_dp')
    from simplegep.trainers.no_dp_trainer import train
    start_train(args, train_fn=train)

def run_dp_sgd_keypressemg():
    args = parse_args(data_name='keypressemg', dp_method='dp_sgd')
    from simplegep.trainers.dp_sgd_trainer import train
    start_train(args, train_fn=train)

def sweep_no_dp_keypressemg():
    config_yaml_path = 'simplegep/sweepers/sweep_configurations/sgd_dp_bayes.yaml'
    sweep_configuration, args, logger = prepare_sweep(data_name='keypressemg', dp_method='no_dp',
                                                      config_yaml_path=config_yaml_path)
    from simplegep.trainers.no_dp_trainer import train

    sweep(sweep_config=sweep_configuration, args=args,
          train_fn=partial(train, logger=logger))

def sweep_dp_sgd_keypressemg():
    config_yaml_path = 'simplegep/sweepers/sweep_configurations/sgd_dp_bayes.yaml'
    sweep_configuration, args, logger = prepare_sweep(data_name='keypressemg', dp_method='dp_sgd',
                                                      config_yaml_path=config_yaml_path)
    from simplegep.trainers.dp_sgd_trainer import train

    sweep(sweep_config=sweep_configuration, args=args,
          train_fn=partial(train, logger=logger))


def main():
    run_no_dp_cifar10()


if __name__ == "__main__":
    # sweep_gep_putemg()
    # run_no_dp_putemg()
    # run_dp_sgd_putemg()
    # run_no_dp_keypressemg()
    run_dp_sgd_keypressemg()


