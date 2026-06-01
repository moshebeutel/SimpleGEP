import argparse
import logging
import random
import time
from pathlib import Path
import numpy as np
import torch

from simplegep.trainers.utils import str2bool

def add_arguments_keypressemg(parser, project_dir: Path):
    parser.add_argument("--depth_power", type=int, default=1)
    parser.add_argument("--num-features", type=int, default=128, choices=[128],
                        help="Number of extracted features (model input size)")
    parser.add_argument("--num-features-per-channel", type=int, default=16,
                        help="Number of extracted features per channel")
    parser.add_argument("--data_root", type=str,
                            # default=(project_dir / 'data/EMG/keypressemg/CleanData/valid_features').as_posix(),
                            default=(project_dir / 'data/EMG/keypressemg/CleanData/valid_features_long_npy').as_posix(),
                            help="dir path for dataset")
    parser.add_argument("--num_classes", type=int, default=26, help="number of classes in the dataset")

    parser.add_argument('--log_data_statistics', type=str, default=True)

    return parser

def add_arguments_putemg(parser, project_dir):
    parser.add_argument("--depth_power", type=int, default=1)
    parser.add_argument("--num-features", type=int, default=128, choices=[384, 128],
                        help="Number of extracted features (model input size)")
    parser.add_argument("--num-features-per-channel", type=int, default=16,
                        help="Number of extracted features per channel")
    parser.add_argument("--data_root", type=str,
                        # default='./data/EMG/putEMG/Data-HDF5-Features-NoArgs',
                        default=(project_dir / 'data/EMG/putEMG/Data-HDF5-Features-Short-Time').as_posix(),
                        # default='./data/EMG/putEMG/Data-HDF5-Features-Small',
                        # default=(Path.home() / 'datasets/EMG/putEMG/Data-HDF5-Features-Small').as_posix(),
                        help="dir path for dataset")
    parser.add_argument("--num_classes", type=int, default=4, help="number of classes in the dataset")

    parser.add_argument('--log_data_statistics', type=str, default=False)

def parse_args(data_name: str, dp_method: str):
    use_gp = False
    session_name = f"{'GP_' if use_gp else ''}{data_name.upper()}_DP_{dp_method.upper()}"
    parser = argparse.ArgumentParser(description=session_name)
    project_dir = Path(__file__).resolve().parent
    model_name = 'resnet20' if data_name == 'cifar10' else 'feature_model'
    # model_name = 'tiny_cifar_net_4'
    ## general arguments
    parser.add_argument('--dataset', default=data_name, type=str, help='dataset name')
    parser.add_argument('--log_root', default=project_dir / 'log', type=str, help='log directory')
    parser.add_argument('--log_level', default='DEBUG', type=str, choices=['DEBUG', 'INFO'],
                        help='log level: DEBUG, INFO Default: DEBUG.')
    parser.add_argument('--use-gp', type=str2bool, default=use_gp)

    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--sess', default=session_name, type=str, help='session name')
    parser.add_argument('--checkpoint', default=f'{model_name}_{data_name}.tar', type=str, help='session name')
    parser.add_argument('--model_name', default=model_name, type=str, help='model name')
    parser.add_argument('--loss_function', default='cross_entropy', type=str, help='loss function name')
    parser.add_argument('--optimizer', default='adam', type=str, help='optimizer name')
    parser.add_argument('--seed', default=2, type=int, help='random seed')
    parser.add_argument('--weight_decay', default=0., type=float, help='weight decay')
    parser.add_argument('--batchsize', default=256, type=int, help='batch size')
    parser.add_argument('--num_epochs', default=200, type=int, help='total number of epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='base learning rate (default=0.1)')
    parser.add_argument('--momentum', default=0.9, type=float, help='value of momentum')

    ## arguments for learning with differential privacy
    parser.add_argument('--dp_method', default=dp_method, choices=['no_dp', 'dp_sgd', 'gep', 'super'],
                        help='Differential privacy method: dp_sgd, gep, no dp, super. Default: dp_sgd.')
    parser.add_argument('--private', '-p', action='store_true', help='enable differential privacy')
    parser.add_argument('--dynamic_noise', action='store_true', help='varying noise levels for each epoch')
    parser.add_argument('--dynamic_noise_high_factor', default=3.2, type=float,
                        help='highest noise factor for varying mechanism')
    parser.add_argument('--dynamic_noise_low_factor', default=0.4, type=float,
                        help='lowest noise factor for varying mechanism')
    parser.add_argument('--decrease_shape', default='step', type=str,
                        choices=['linear', 'geometric', 'logarithmic', 'step'])

    parser.add_argument('--clip_strategy', default='median', type=str, choices=['value', 'median', 'max'],
                        help='clip strategy name: value, median, max')
    parser.add_argument('--clip_value', default=5., type=float, help='gradient clipping bound')
    parser.add_argument('--eps', default=8., type=float, help='privacy parameter epsilon')
    parser.add_argument('--dp_sigma', default=0., type=float, help='privacy noise factor')

    ## arguments for GEP
    parser.add_argument('--embedder', default='svd', type=str, choices=['svd', 'kernel_pca'],
                        help='embedder name for GEP')
    parser.add_argument('--kernel_type', default='rbf', type=str,
                        choices=["linear", "rbf", "poly", "sigmoid", "cosine"], help='embedder name for GEP')
    parser.add_argument('--num_basis', default=1000, type=int, help='total number of basis elements')
    parser.add_argument('--grads_history_size', default=1000, type=int,
                        help='total number of history grads to keep for basis calculation')
    parser.add_argument('--stop_embedding_epoch', default=1e10, type=int, help='switch to dp sgd after that epoch')

    parser.add_argument('--real_labels', action='store_true', help='use real labels for auxiliary dataset')
    parser.add_argument('--aux_dataset', default='imagenet', type=str,
                        help='name of the public dataset, [cifar10, cifar100, imagenet]')
    parser.add_argument('--aux_data_size', default=2000, type=int, help='size of the auxiliary dataset')
    parser.add_argument('--wandb', type=bool, default=True, help='enable wandb')

    if (data_name == 'putemg'):
        add_arguments_putemg(parser, project_dir=project_dir)
    elif (data_name == 'keypressemg'):
        add_arguments_keypressemg(parser, project_dir=project_dir)
    else:
        parser.add_argument('--data_root', default=project_dir / 'data', type=str, help='dataset directory')

    args = parser.parse_args()
    return args


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


