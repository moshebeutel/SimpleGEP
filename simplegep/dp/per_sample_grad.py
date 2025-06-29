import torch
from backpack import extend, backpack
from backpack.extensions import BatchGrad

from simplegep.dp.utils import flatten_tensor


def pretrain_actions(model, loss_func):
    model = extend(model)
    loss_func = extend(loss_func)
    return model, loss_func


def backward_pass_get_batch_grads(batch_loss: torch.Tensor, net: torch.nn.Module) -> torch.Tensor:
    grad_batch_list = []
    with backpack(BatchGrad()):
        batch_loss.backward()
    for p in net.parameters():
        grad_batch_list.append(p.grad_batch.reshape(p.grad_batch.shape[0], -1))
        p.grad_batch = p.grad_batch.detach().cpu()
        p.grad_batch = None
        del p.grad_batch

    flat_grad_batch_tensor = flatten_tensor(grad_batch_list)

    grad_batch_list = [t.detach().cpu() for t in grad_batch_list]
    grad_batch_list = None
    del grad_batch_list

    return flat_grad_batch_tensor
