import torch
from tqdm import tqdm


@torch.no_grad()
def eval_model(net, train_loader, eval_loader, GP):
    net.eval()
    GP, label_map, Y_train, X_train = build_tree(net, train_loader, GP)

    is_first_iter = True
    running_loss, running_correct, running_samples = 0., 0., 0.

    GP.eval()
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

            pbar.set_description(f'Batch {batch_idx}/{len(eval_loader)} eval batch loss {loss.item():.2f}'
                                 f' eval accuracy {batch_acc:.2f}')


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