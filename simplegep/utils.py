import argparse
import logging
import os
import random
import time
from pathlib import Path
import numpy as np
import torch


def parse_args(description: str):
    parser = argparse.ArgumentParser(description=description)

    ## general arguments
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
    parser.add_argument('--data_root', default='data', type=str, help='dataset name')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--sess', default='resnet20_cifar10', type=str, help='session name')
    parser.add_argument('--model_name', default='resnet20', type=str, help='model name')
    parser.add_argument('--loss_function', default='cross_entropy', type=str, help='loss function name')
    parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer name')
    parser.add_argument('--seed', default=2, type=int, help='random seed')
    parser.add_argument('--weight_decay', default=0., type=float, help='weight decay')
    parser.add_argument('--batchsize', default=256, type=int, help='batch size')
    parser.add_argument('--num_epochs', default=10, type=int, help='total number of epochs')
    parser.add_argument('--lr', default=0.01, type=float, help='base learning rate (default=0.1)')
    parser.add_argument('--momentum', default=0.9, type=float, help='value of momentum')

    ## arguments for learning with differential privacy
    parser.add_argument('--private', '-p', action='store_true', help='enable differential privacy')
    parser.add_argument('--clip_strategy', default='median', type=str,
                        help='clip strategy name: value, median, max')
    parser.add_argument('--clip_value', default=5., type=float, help='gradient clipping bound')
    parser.add_argument('--eps', default=8., type=float, help='privacy parameter epsilon')

    args = parser.parse_args()
    return args

def load_checkpoint(checkpoint_path: str, net: torch.nn.Module, optimizer):
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists(), f'Requested checkpoint {checkpoint_path} does not exist'

    checkpoint = torch.load(checkpoint_path, weights_only=True)

    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    acc = checkpoint['acc']
    seed = checkpoint['seed']
    rng_state = checkpoint['rng_state']
    return epoch, acc, seed, rng_state


def save_checkpoint(net, optimizer, acc, epoch, seed, sess):
    state = {
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'seed': seed,
        'rng_state': torch.get_rng_state(),
    }

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + sess + f'epoch_{epoch}_acc_{acc}.ckpt')

def set_seed(seed, cudnn_enabled=True):
    """for reproducibility

    :param seed:
    :return:
    """

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = cudnn_enabled
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_logger(logger_name: str, log_dir: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f'{logger_name}_{time.asctime()}.log')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger