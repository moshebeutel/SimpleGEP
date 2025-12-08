import wandb
from simplegep.utils import parse_args, set_logger, set_seed


def main(args):
    logger = set_logger(logger_name=args.sess, log_dir=args.log_root, level=args.log_level)
    logger.info(f'Logger is set - session: {args.sess}')
    logger.info(f'Arguments: {args}')
    with wandb.init(project='GEP', name=args.sess):
        set_seed(args.seed)
        wandb.run.name = '_'.join([f'{k}_{getattr(args,k)}'.upper() for k in ['dp_method',
                                                                              'model_name',
                                                                              'dataset',
                                                                              'eps', 'dynamic_noise',
                                                                              'optimizer', 'lr', 'batchsize']])
        train(args, logger)


if __name__ == "__main__":
    args = parse_args(description=f'Differentially Private learning')
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
