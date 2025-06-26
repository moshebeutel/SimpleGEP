import torch


optimizer_hub = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam,}

def get_optimizer(optimizer_name, model, lr):
    assert optimizer_name in optimizer_hub, 'Optimizer not found'
    optimizer_func = optimizer_hub[optimizer_name]
    optimizer = optimizer_func(model.parameters(), lr=lr)
    return optimizer
