from simplegep.trainers.dp_sgd_trainer import train
from simplegep.utils import parse_args, set_logger


def main(args):
    logger = set_logger(logger_name=args.sess, log_dir='log', level='DEBUG')
    logger.info(f'Logger is set - session: {args.sess}')
    logger.info(f'Arguments: {args}')
    train(args, logger)


if __name__ == "__main__":
    args = parse_args(description='Differentially Private learning with DP-SGD')
    main(args)
