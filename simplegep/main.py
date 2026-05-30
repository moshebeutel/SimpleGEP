from argparse import Namespace

import wandb
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
        wandb.run.name = '_'.join([f'{k}_{getattr(args, k)}'.upper() for k in ['dp_method',
                                                                               'model_name',
                                                                               'dataset',
                                                                               'eps', 'dynamic_noise',
                                                                               'optimizer', 'lr', 'batchsize']])
        train_fn(args, logger)

def run_dp_sgd_cifar10():
    args = parse_args(data_name='cifar10', dp_method='dp_sgd')
    from simplegep.trainers.dp_sgd_trainer import train
    start_train(args, train_fn=train)

def run_dp_sgd_keypressemg():
    args = parse_args(data_name='keypressemg', dp_method='dp_sgd')
    from simplegep.trainers.dp_sgd_trainer import train
    start_train(args, train_fn=train)


def run_dp_sgd_putemg():
    args = parse_args(data_name='putemg', dp_method='dp_sgd')
    from simplegep.trainers.dp_sgd_trainer import train
    start_train(args, train_fn=train)

def run_no_dp_keypressemg():
    args = parse_args(data_name='keypressemg', dp_method='no_dp')
    from simplegep.trainers.no_dp_trainer import train
    start_train(args, train_fn=train)



def run_no_dp_putemg():
    args = parse_args(data_name='putemg', dp_method='no_dp')
    from simplegep.trainers.no_dp_trainer import train
    start_train(args, train_fn=train)


def run_no_dp_cifar10():
    args = parse_args(data_name='cifar10', dp_method='no_dp')
    from simplegep.trainers.no_dp_trainer import train
    start_train(args, train_fn=train)





def main():
    run_no_dp_cifar10()


if __name__ == "__main__":
    # run_no_dp_putemg()
    # run_dp_sgd_putemg()
    run_no_dp_keypressemg()
    # run_dp_sgd_keypressemg()
