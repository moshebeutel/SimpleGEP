import argparse
import gc
import torch
from tqdm import tqdm
from simplegep.dp.per_sample_grad import backward_pass_get_batch_grads
from simplegep.models.utils import substitute_grads

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
import gc
import torch


def clear_cuda_from_namespace(namespace: dict, verbose: bool = True):
    cleared = []

    for name, value in list(namespace.items()):
        if name.startswith("__"):
            continue

        try:
            if torch.is_tensor(value) and value.is_cuda:
                namespace[name] = None
                cleared.append(name)

            elif isinstance(value, torch.nn.Module):
                value.zero_grad(set_to_none=True)
                value.cpu()
                namespace[name] = None
                cleared.append(name)

            elif isinstance(value, torch.optim.Optimizer):
                for group in value.param_groups:
                    for param in group.get("params", []):
                        if param is not None:
                            param.grad = None

                value.state.clear()
                namespace[name] = None
                cleared.append(name)

        except Exception:
            pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if verbose:
        print(f"Cleared names: {cleared}")

    return cleared
@torch.no_grad()
def eval_model(net, loss_function, loader):
    net.eval()
    eval_loss = 0
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

            eval_loss += step_loss
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct_idx = predicted.eq(targets.data).cpu()
            all_correct += correct_idx.numpy().tolist()
            correct += correct_idx.sum()
            batch_acc = correct_idx.sum() / targets.size(0)

            pbar.set_description(f'Batch {batch_idx}/{len(loader)} eval batch loss {step_loss:.2f}'
                                 f' eval accuracy {batch_acc:.2f}')

            inputs, targets, outputs, loss = (inputs.detach().cpu(), targets.detach().cpu(),
                                              outputs.detach().cpu(), loss.detach().cpu())
            inputs, targets, outputs, loss = None, None, None, None
            del inputs, targets, outputs, loss
            gc.collect()
            torch.cuda.empty_cache()

        eval_acc = 100. * float(correct) / float(total)
        eval_loss = eval_loss / batch_idx

    return eval_loss, eval_acc

def train_epoch(net, loss_function, optimizer, train_loader, grads_processor):
    train_loss, train_acc, correct, total, batch_idx = 0.0, 0.0, 0, 0, 0
    all_correct = []
    net.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()

        # forward pass
        outputs = net(inputs)
        loss = loss_function(outputs, targets)
        step_loss = loss.item()
        step_loss /= inputs.shape[0]
        train_loss += step_loss
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct_idx = predicted.eq(targets.data).cpu()
        all_correct += correct_idx.numpy().tolist()
        correct += correct_idx.sum()
        batch_acc = correct_idx.sum() / targets.size(0)

        # get per sample grads
        flat_per_sample_grads = backward_pass_get_batch_grads(batch_loss=loss, net=net)

        # perturb grads
        processed_grads = grads_processor.process_grads(flat_per_sample_grads).squeeze()

        # substitute perturbed grads
        substitute_grads(net, processed_grads)

        # update net parameters
        optimizer.step()

        pbar.set_description(f'Batch {batch_idx}/{len(train_loader)} train batch loss {step_loss:.2f}'
                             f' train accuracy {batch_acc:.2f}')

        # free gpu memory
        inputs, targets, outputs, loss = (inputs.detach().cpu(), targets.detach().cpu(),
                                          outputs.detach().cpu(), loss.detach().cpu())
        inputs, targets, outputs, loss = None, None, None, None
        del inputs, targets, outputs, loss
        gc.collect()
        torch.cuda.empty_cache()

    train_acc = 100. * float(correct) / float(total)
    train_loss = train_loss / batch_idx

    return train_loss, train_acc
