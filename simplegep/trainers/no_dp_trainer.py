import gc
import logging
import torch
import wandb
from tqdm import tqdm
from simplegep.data.cifar_loader import get_train_loader, get_test_loader
from simplegep.dp.per_sample_grad import pretrain_actions
from simplegep.models.factory import get_model
from simplegep.models.utils import initialize_weights, count_parameters
from simplegep.trainers.utils import eval_model
from simplegep.trainers.factory import get_loss_function, get_optimizer


def train_epoch(net, loss_function, optimizer, train_loader):
    train_loss, train_acc = 0.0, 0.0
    correct = 0
    total = 0
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

        # backward pass
        loss.backward()

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


def train(args, logger: logging.Logger):
    logger.info('Starting training')
    if args.wandb:
        wandb.init(project='GEP', name=args.sess)

    net = get_model(args)
    initialize_weights(net)
    num_params, layer_sizes = count_parameters(model=net, return_layer_sizes=True)
    logger.debug(f'Model set to {args.model_name} num params {num_params}')
    logger.debug(f'layer sizes: {layer_sizes}')

    # reduction = 'sum' if args.private else 'mean'
    reduction = 'sum'
    loss_function = get_loss_function(args.loss_function, reduction=reduction)
    logger.debug(f'loss function set to {args.loss_function} reduction {reduction}')

    net, loss_function = pretrain_actions(model=net, loss_func=loss_function)
    logger.debug('model and loss functions prepared for per sample grads')

    optimizer = get_optimizer(args=args, model=net)
    logger.debug(f'optimizer set to {args.optimizer} lr {args.lr}')

    train_loader = get_train_loader(root=args.data_root, batchsize=args.batchsize)
    logger.debug(f'train loader created size {len(train_loader)}')
    test_loader = get_test_loader(root=args.data_root, batchsize=args.batchsize)
    logger.debug(f'test loader created size {len(test_loader)}')

    num_epochs = args.num_epochs
    net = net.cuda()
    for epoch in range(num_epochs):
        logger.info(f'***** Starting epoch {epoch}  ******')
        train_loss, train_acc = train_epoch(net=net, loss_function=loss_function, optimizer=optimizer,
                                            train_loader=train_loader)
        logger.info(f'Epoch {epoch}/{args.num_epochs} train loss {train_loss:.2f} train accuracy {train_acc:.2f}')
        test_loss, test_acc = eval_model(net=net, loss_function=loss_function, loader=test_loader)
        logger.info(f'Epoch {epoch}/{args.num_epochs} test loss {test_loss:.2f} test accuracy {test_acc:.2f}')
        if args.wandb:
            wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'test_loss': test_loss,
                       'test_acc': test_acc}, step=epoch)

