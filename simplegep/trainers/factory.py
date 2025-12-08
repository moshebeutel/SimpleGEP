import torch

loss_function_hub = {'cross_entropy': torch.nn.CrossEntropyLoss}


def get_loss_function(loss_function_name, reduction):
    assert loss_function_name in loss_function_hub, 'Loss function not found'
    loss_function_ctor = loss_function_hub[loss_function_name]
    loss_function = loss_function_ctor(reduction=reduction)
    return loss_function


optimizer_hub = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam}


def get_optimizer(args, model):
    assert args.optimizer in optimizer_hub, 'Optimizer not found'
    optimizer_func = optimizer_hub[args.optimizer]
    param_keys = ['lr', 'weight_decay'] if args.optimizer == 'adam' else ['lr', 'weight_decay', 'momentum']
    param_dict = {key:args.__dict__[key] for key in param_keys}
    optimizer = optimizer_func(model.parameters(), **param_dict)
    return optimizer
