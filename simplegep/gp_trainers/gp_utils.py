import copy
from collections import defaultdict
from typing import Dict
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
# from fed_trainers.trainers.utils import detach_to_numpy, get_device
# from fed_trainers.trainers.factory import get_optimizer, get_logger


@torch.no_grad()
def eval_model(net, train_loader, eval_loader, GP):
    net.eval()
    eval_loss = 0
    correct = 0
    total = 0
    all_correct = []
    GP, label_map, Y_train, X_train = build_tree(net, train_loader, GP)

    # # results: defaultdict[int, defaultdict[str, float]] = defaultdict()
    # results = defaultdict(lambda: defaultdict(list))

    targets = []
    preds = []
    step_results = []
    is_first_iter = True
    running_loss, running_correct, running_samples = 0., 0., 0.


    # build tree at each step

    GP.eval()
    # data_labels = []
    # data_preds = []
    with torch.no_grad():
        pbar = tqdm(enumerate(eval_loader), total=len(eval_loader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.cuda(), targets.cuda()
            Y_test = torch.tensor([label_map[l.item()] for l in targets], dtype=targets.dtype,
                                  device=targets.device)

            X_test = net(inputs)
            loss, pred = GP.forward_eval(X_train, Y_train, X_test, Y_test, is_first_iter)
            batch_size = Y_test.shape[0]
            running_loss += (loss.item() * batch_size)
            batch_correct = pred.argmax(1).eq(Y_test).sum().item()
            running_correct += batch_correct
            running_samples += batch_size
            batch_acc = batch_correct / batch_size

            is_first_iter = False
            targets.append(Y_test)
            preds.append(pred)

            # data_labels.append(Y_test)
            # data_preds.append(pred)
            pbar.set_description(f'Batch {batch_idx}/{len(eval_loader)} eval batch loss {loss.item():.2f}'
                                 f' eval accuracy {batch_acc:.2f}')

    # # calculate confusion matrix
    # cm = confusion_matrix(detach_to_numpy(torch.cat(data_labels, dim=0)),
    #                       detach_to_numpy(torch.max(torch.cat(data_preds, dim=0), dim=1)[1]))
    #
    # # save classification results to output structure
    # step_results.append({"cm": cm,
    #                      "y_true": detach_to_numpy(torch.cat(data_labels, dim=0)),
    #                      "y_pred": detach_to_numpy(torch.max(torch.cat(data_preds, dim=0), dim=1)[1])})
    #
    # # erase tree (no need to save it)
    # GP.tree = None
    #
    # results['loss'] = running_loss / running_samples
    # results['correct'] = running_correct
    # results['total'] = running_samples
    #
    # target = detach_to_numpy(torch.cat(targets, dim=0))
    # full_pred = detach_to_numpy(torch.cat(preds, dim=0))
    # labels_vs_preds = np.concatenate((target.reshape(-1, 1), full_pred), axis=1)
    #
    # # =============================
    # # GLOBAL TOTALS (across clients)
    # # =============================
    # # Concatenate all true labels and prediction scores
    # y_true_all_t = torch.cat(targets, dim=0)  # tensor on device
    # y_prob_all_t = torch.cat(preds, dim=0)  # tensor on device
    #
    # # Convert to numpy
    # y_true_all = detach_to_numpy(y_true_all_t)  # shape: [N]
    # y_prob_all = detach_to_numpy(y_prob_all_t)  # shape: [N, C]
    # y_pred_all = y_prob_all.argmax(axis=1)  # shape: [N]

    eval_loss = running_loss / running_samples
    eval_acc = 100. * running_correct / running_samples


    # return results, labels_vs_preds, step_results, y_true_all, y_pred_all
    return eval_loss, eval_acc

@torch.no_grad()
def build_tree(net, loader, GP):
    """
    Build GP tree per client
    :return: List of GPs
    """
    for k, batch in enumerate(loader):
        batch = (t.cuda() for t in batch)
        train_data, clf_labels = batch

        z = net(train_data)
        X = torch.cat((X, z), dim=0) if k > 0 else z
        Y = torch.cat((Y, clf_labels), dim=0) if k > 0 else clf_labels

    # build label map
    client_labels, client_indices = torch.sort(torch.unique(Y))
    label_map = {client_labels[i].item(): client_indices[i].item() for i in range(client_labels.shape[0])}
    offset_labels = torch.tensor([label_map[l.item()] for l in Y], dtype=Y.dtype,
                                 device=Y.device)

    GP.build_base_tree(X, offset_labels)  # build tree
    return GP, label_map, offset_labels, X



def add_arguments_gp(parser):
    # parser.add_argument("--n-kernels", type=int, default=16, help="number of kernels")

    # parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--loss-scaler', default=1., type=float, help='multiplicative element to the loss function')
    parser.add_argument('--kernel-function', type=str, default='RBFKernel',
                        choices=['RBFKernel', 'LinearKernel', 'MaternKernel'],
                        help='kernel function')
    parser.add_argument('--objective', type=str, default='predictive_likelihood',
                        choices=['predictive_likelihood', 'marginal_likelihood'])
    parser.add_argument('--predict-ratio', type=float, default=0.5,
                        help='ratio of samples to make predictions for when using predictive_likelihood objective')
    parser.add_argument('--num-gibbs-steps-train', type=int, default=5, help='number of sampling iterations')
    parser.add_argument('--num-gibbs-draws-train', type=int, default=20, help='number of parallel gibbs chains')
    parser.add_argument('--num-gibbs-steps-test', type=int, default=5, help='number of sampling iterations')
    parser.add_argument('--num-gibbs-draws-test', type=int, default=30, help='number of parallel gibbs chains')
    parser.add_argument('--outputscale', type=float, default=8., help='output scale')
    parser.add_argument('--lengthscale', type=float, default=1., help='length scale')
    parser.add_argument('--outputscale-increase', type=str, default='constant',
                        choices=['constant', 'increase', 'decrease'],
                        help='output scale increase/decrease/constant along tree')

    return parser