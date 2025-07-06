import torch
from torch import nn
from backpack import extend, backpack
from backpack.extensions import BatchGrad
from torch.utils.data import DataLoader, TensorDataset
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

class PublicDataPerSampleGrad:
    def __init__(self,  public_data, net:nn.Module,public_batchsize: int = 256):
        if isinstance(public_data, DataLoader):
            self._public_data_loader = public_data
        elif isinstance(public_data, tuple):
            assert (len(public_data) == 2 and
                    isinstance(public_data[0], torch.Tensor) and
                    isinstance(public_data[1], torch.Tensor)), 'public_data must be a tuple of two tensors'


            dataset = TensorDataset(*public_data)
            self._public_data_loader = DataLoader(dataset, batch_size=public_batchsize, shuffle=True,
                                                                   num_workers=2, drop_last=True)
        else:
            raise ValueError('public_data must be either torch.Tensor or torch.utils.data.DataLoader')
        self._net = net
    def get_grads(self, current_state_dict: dict):
        self._net.load_state_dict(current_state_dict)
        grad_batch_list = []
        for batch in self._public_data_loader:
            batch = batch[0].cuda()
            batch_loss = self._net(batch).mean()
            grad_batch = backward_pass_get_batch_grads(batch_loss, self._net)
            grad_batch_list.append(grad_batch)
        flat_grad_batch_tensor = torch.cat(grad_batch_list)
        return flat_grad_batch_tensor