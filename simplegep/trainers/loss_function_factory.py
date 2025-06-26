import torch

loss_function_hub = {'cross_entropy': torch.nn.CrossEntropyLoss}

def get_loss_function(loss_function_name, reduction):
    assert loss_function_name in loss_function_hub, 'Loss function not found'
    loss_function_ctor = loss_function_hub[loss_function_name]
    loss_function = loss_function_ctor(reduction=reduction)
    return loss_function