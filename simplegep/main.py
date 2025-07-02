from simplegep.utils import parse_args, set_logger

def main(args):
    logger = set_logger(logger_name=args.sess, log_dir=args.log_directory, level=args.log_level)
    logger.info(f'Logger is set - session: {args.sess}')
    logger.info(f'Arguments: {args}')
    train(args, logger)


if __name__ == "__main__":
    args = parse_args(description=f'Differentially Private learning')
    if args.dp_method == 'gep':
        from simplegep.trainers.gep_trainer import train
    else:
        assert args.dp_method == 'dp_sgd'
        from simplegep.trainers.dp_sgd_trainer import train

    main(args)
