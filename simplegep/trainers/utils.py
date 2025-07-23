import gc

import torch
from tqdm import tqdm


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

        test_acc = 100. * float(correct) / float(total)
        test_loss = test_loss / batch_idx

    return test_loss, test_acc