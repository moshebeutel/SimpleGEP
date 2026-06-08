import os
from pathlib import Path

import torch
from torch import nn


def initialize_weights(module: nn.Module):
    for m in module.modules():

        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()


def count_parameters(model, return_layer_sizes=False):
    relevant_params = [p for (n,p) in model.named_parameters() if p.requires_grad and 'batch_norm' not in n]
    layer_sizes = [p.numel() for p in relevant_params]
    sum_params = sum(layer_sizes)
    return sum_params if not return_layer_sizes else sum_params, layer_sizes



def substitute_grads(net, grads):
    offset = 0
    for n,param in net.named_parameters():
        if '_batch_norm' in n:
            continue
        numel = param.numel()
        grad = grads[offset:offset + numel].reshape(param.shape).to(param.device)
        param.grad = grad.clone().reshape(param.shape)
        offset += numel


def load_checkpoint(checkpoint_path: str, net: torch.nn.Module, optimizer=None):
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists(), f'Requested checkpoint {checkpoint_path} does not exist'

    checkpoint = torch.load(checkpoint_path, weights_only=True)

    net.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
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
    checkpoint_name = './checkpoint/' + sess + f'_epoch_{epoch}_acc_{acc}.tar'
    torch.save(state, checkpoint_name)
    return checkpoint_name