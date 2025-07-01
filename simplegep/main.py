from simplegep.utils import parse_args, set_logger
DP_METHOD = 'DP-SGD'
# DP_METHOD = 'GEP'
if DP_METHOD == 'GEP':
    from simplegep.trainers.gep_trainer import train
else:
    from simplegep.trainers.dp_sgd_trainer import train

def main(args):
    logger = set_logger(logger_name=args.sess, log_dir='log', level='DEBUG')
    logger.info(f'Logger is set - session: {args.sess}')
    logger.info(f'Arguments: {args}')
    train(args, logger)


if __name__ == "__main__":
    args = parse_args(description=f'Differentially Private learning with {DP_METHOD}')
    main(args)
