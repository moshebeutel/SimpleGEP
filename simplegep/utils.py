import argparse
import gc
import logging
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm


def parse_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    project_dir = Path(__file__).resolve().parent
    ## general arguments
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
    parser.add_argument('--data_root', default= project_dir / 'data', type=str, help='dataset directory')
    parser.add_argument('--log_root', default= project_dir / 'log', type=str, help='log directory')
    parser.add_argument('--log_level', default= 'DEBUG', type=str, choices=['DEBUG', 'INFO'],
                        help='log level: DEBUG, INFO Default: DEBUG.')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--sess', default='resnet20_cifar10', type=str, help='session name')
    parser.add_argument('--model_name', default='tiny_cifar_net_4', type=str, help='model name')
    parser.add_argument('--loss_function', default='cross_entropy', type=str, help='loss function name')
    parser.add_argument('--optimizer', default='adam', type=str, help='optimizer name')
    parser.add_argument('--seed', default=2, type=int, help='random seed')
    parser.add_argument('--weight_decay', default=0., type=float, help='weight decay')
    parser.add_argument('--batchsize', default=256, type=int, help='batch size')
    parser.add_argument('--num_epochs', default=30, type=int, help='total number of epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='base learning rate (default=0.1)')
    parser.add_argument('--momentum', default=0.9, type=float, help='value of momentum')

    ## arguments for learning with differential privacy
    parser.add_argument('--dp_method', default="dp_sgd", choices=['dp_sgd', 'gep'],
                        help='Differential privacy method: dp_sgd, gep. Default: dp_sgd.')
    parser.add_argument('--private', '-p', action='store_true', help='enable differential privacy')
    parser.add_argument('--dynamic_noise', action='store_true', help='varying noise levels for each epoch')
    parser.add_argument('--dynamic_noise_high_factor', default=10., type=float, help='highest noise factor for varying mechanism')
    parser.add_argument('--dynamic_noise_low_factor', default=0.1, type=float, help='lowest noise factor for varying mechanism')
    parser.add_argument('--decrease_shape', default='linear', type=str, choices=['linear', 'geometric', 'logarithmic'])

    parser.add_argument('--clip_strategy', default='median', type=str, choices=['value', 'median', 'max'],
                        help='clip strategy name: value, median, max')
    parser.add_argument('--clip_value', default=5., type=float, help='gradient clipping bound')
    parser.add_argument('--eps', default=8., type=float, help='privacy parameter epsilon')

    ## arguments for GEP
    parser.add_argument('--embedder', default='svd', type=str, choices=['svd', 'kernel_pca'], help='embedder name for GEP')
    parser.add_argument('--kernel_type', default='rbf', type=str, choices=["linear", "rbf", "poly", "sigmoid", "cosine"], help='embedder name for GEP')
    parser.add_argument('--num_basis', default=1000, type=int, help='total number of basis elements')
    parser.add_argument('--grads_history_size', default=1000, type=int, help='total number of history grads to keep for basis calculation')

    parser.add_argument('--real_labels', action='store_true', help='use real labels for auxiliary dataset')
    parser.add_argument('--aux_dataset', default='imagenet', type=str,
                        help='name of the public dataset, [cifar10, cifar100, imagenet]')
    parser.add_argument('--aux_data_size', default=2000, type=int, help='size of the auxiliary dataset')
    parser.add_argument('--wandb', type=bool, default=True, help='enable wandb')

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


@torch.no_grad()
def eval_model(net, loss_function, loader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_correct = []
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=len(loader))
        for batch_idx, (inputs, targets) in pbar:

            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = loss_function(outputs, targets)
            step_loss = loss.item()

            step_loss /= inputs.shape[0]

            test_loss += step_loss
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct_idx = predicted.eq(targets.data).cpu()
            all_correct += correct_idx.numpy().tolist()
            correct += correct_idx.sum()
            batch_acc = correct_idx.sum() / targets.size(0)

            pbar.set_description(f'Batch {batch_idx}/{len(loader)} test batch loss {step_loss:.2f}'
                                 f' test accuracy {batch_acc:.2f}')

            inputs, targets, outputs, loss = (inputs.detach().cpu(), targets.detach().cpu(),
                                                               outputs.detach().cpu(), loss.detach().cpu())
            inputs, targets, outputs, loss = None, None, None, None
            del inputs, targets, outputs, loss
            gc.collect()
            torch.cuda.empty_cache()

        test_acc = 100.*float(correct)/float(total)
        test_loss = test_loss / batch_idx

    return test_loss, test_acc
