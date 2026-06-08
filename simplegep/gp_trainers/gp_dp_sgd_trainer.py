import logging
import torch
import wandb
from tqdm import tqdm
import gc

from pFedGP.pFedGP.Learner import pFedGPFullLearner

from simplegep.dp.dp_params import get_dp_params
from simplegep.dp.grads_proc import GradsProcessor
from simplegep.dp.per_sample_grad import pretrain_actions, backward_pass_get_batch_grads
from simplegep.gp_trainers.gp_utils import build_tree, eval_model
from simplegep.models.factory import get_model
from simplegep.models.utils import initialize_weights, count_parameters, load_checkpoint, save_checkpoint, \
    substitute_grads
from simplegep.trainers.dp_sgd_trainer import compute_dynamic_dp_params
from simplegep.trainers.factory import get_optimizer, get_dataloaders


def train_epoch(net,
                optimizer,
                train_loader, grads_processor,
                GP):

    # build tree at each step
    net.train()
    GP, label_map, _, __ = build_tree(net, train_loader, GP)
    GP.train()
    optimizer.zero_grad()
    running_loss, running_correct, running_samples = 0., 0., 0.
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.cuda(), targets.cuda()

        # forward prop
        outputs = net(inputs)

        X = torch.cat((X, outputs), dim=0) if batch_idx > 0 else outputs
        Y = torch.cat((Y, targets), dim=0) if batch_idx > 0 else targets

        running_samples += targets.shape[0]

        # free gpu memory
        inputs, targets, outputs = (inputs.detach().cpu(), targets.detach().cpu(), outputs.detach().cpu())
        inputs, targets, outputs = None, None, None
        del inputs, targets, outputs
        gc.collect()
        torch.cuda.empty_cache()

        pbar.set_description(f'Batch {batch_idx}/{len(train_loader)}')

    offset_labels = torch.tensor([label_map[l.item()] for l in Y], dtype=Y.dtype,
                                 device=Y.device)

    loss = GP(X, offset_labels, to_print=1)
    # loss *= args.loss_scaler
    running_loss += loss.item() * offset_labels.shape[0]

    flat_per_sample_grads = backward_pass_get_batch_grads(batch_loss=loss, net=net)

    # perturb grads
    processed_grads = grads_processor.process_grads(flat_per_sample_grads).squeeze()

    # substitute perturbed grads
    substitute_grads(net, processed_grads)

    optimizer.step()


    # free gpu memory
    offset_labels, processed_grads, flat_per_sample_grads, loss = (offset_labels.detach().cpu(),
                                                                   processed_grads.detach().cpu(),
                                                                   flat_per_sample_grads.detach().cpu(),
                                                                   loss.detach().cpu())
    offset_labels, processed_grads, flat_per_sample_grads, loss = None, None, None, None
    del  offset_labels, processed_grads, flat_per_sample_grads, loss
    gc.collect()
    torch.cuda.empty_cache()

    train_loss = running_loss / running_samples

    return train_loss


def train(args, logger: logging.Logger):
    logger.info(f'Starting training {__file__}')
    GP = pFedGPFullLearner(args, args.num_classes)
    net = get_model(args)
    initialize_weights(net)
    num_params, layer_sizes = count_parameters(model=net, return_layer_sizes=True)
    logger.debug(f'Model set to {args.model_name} num params {num_params}')
    logger.debug(f'layer sizes: {layer_sizes}')

    best_val_acc = 0.0
    start_epoch = 0
    checkpoint_name = ''
    if args.resume:
        start_epoch, best_val_acc, seed, rng_state = load_checkpoint(checkpoint_path=args.checkpoint, net=net,
                                                                 optimizer=None)
        assert args.seed == seed, f'Expected checkpoint seed equals session seed. Got {seed} != {args.seed}'
        logger.info(f'Loaded checkpoint {args.checkpoint} with epoch {start_epoch} best acc {best_val_acc}')

    net = pretrain_actions(model=net)
    logger.debug('model and loss functions prepared for per sample grads')
    net = net.cuda()

    optimizer = get_optimizer(args=args, model=net)
    logger.debug(f'optimizer set to {args.optimizer} lr {args.lr}')

    train_loader, val_loader, test_loader = get_dataloaders(args)

    logger.debug(f'train loader created size {len(train_loader)}')
    logger.debug(f'test loader created size {len(test_loader)}')



    dp_params = get_dp_params(batchsize=args.batchsize,
                              num_training_samples=len(train_loader.dataset),
                              num_epochs=args.num_epochs,
                              epsilon=args.eps, sigma=args.dp_sigma)

    logger.info(f'DP params - '
                 f' sigma {dp_params.sigma}'
                 f' delta {dp_params.delta} '
                 f' epsilon {dp_params.epsilon}'
                 f' sampling prob {dp_params.sampling_prob} '
                 f' steps {dp_params.steps} ')

    sigma_list = [dp_params.sigma] * args.num_epochs
    if args.dynamic_noise:
        accumulated_epsilon_bar_list, accumulated_epsilon_list, sigma_list, sigma_orig, sigma_decrease_function_name = compute_dynamic_dp_params(args,
                                                                                                                   dp_params,
                                                                                                                   start_epoch)

        logger.debug(f'Using decrease function {sigma_decrease_function_name}')

        logger.info(f'Created varying sigma list with {len(sigma_list)} values')
        logger.debug(f'Sigma list: {sigma_list}')
        logger.debug(f'Accumulated epsilon list: {accumulated_epsilon_list}')
        logger.debug(f'Accumulated epsilon bar list: {accumulated_epsilon_bar_list}')
        logger.debug(f'Sigma orig: {sigma_orig}')

    assert args.num_epochs > start_epoch, f'Expected num epochs > start epoch. Got {args.num_epochs} <= {start_epoch}'
    num_epochs = min(args.num_epochs, start_epoch + len(sigma_list)) if args.dynamic_noise else args.num_epochs
    # num_epochs = min(args.num_epochs - start_epoch, len(sigma_list)) if args.dynamic_noise else args.num_epochs
    # assert num_epochs > start_epoch, f'Expected num epochs > start epoch. Got {num_epochs} <= {start_epoch}'

    grads_processor = GradsProcessor(clip_strategy_name=args.clip_strategy,
                                     noise_multiplier=sigma_list,
                                     clip_value=args.clip_value)

    logger.debug(f'Created GradsProcessor with strategy {args.clip_strategy} '
                 f'noise multiplier {dp_params.sigma}'
                 f' clip value {args.clip_value}')

    for epoch in range(start_epoch, num_epochs):
        logger.info(f'***** Starting epoch {epoch}  ******')
        train_loss = train_epoch(net=net, optimizer=optimizer,
                                            train_loader=train_loader, grads_processor=grads_processor, GP=GP)
        logger.info(f'Epoch {epoch}/{args.num_epochs} train loss {train_loss:.2f}')
        val_loss, val_acc = eval_model(net=net, train_loader=train_loader, eval_loader=val_loader, GP=GP)
        logger.info(f'Epoch {epoch}/{args.num_epochs} val loss {val_loss:.2f} val accuracy {val_acc:.2f}')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_name = save_checkpoint(net=net,
                                              optimizer=optimizer,
                                              acc=val_acc,
                                              epoch=epoch,
                                              seed=args.seed,
                                              sess=args.sess)
            logger.info(f'Best Acc = {best_val_acc}. Checkpoint {checkpoint_name} saved!')
        if args.wandb:
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss,
                       'val_acc': val_acc, 'best_val_acc': best_val_acc,
                       'sigma': sigma_list[epoch-start_epoch]}, step=epoch)
            if args.dynamic_noise:
                wandb.log({'accumulated_epsilon': accumulated_epsilon_list[epoch-start_epoch],
                           'accumulated_epsilon_bar': accumulated_epsilon_bar_list[epoch-start_epoch]}, step=epoch)
            # if args.wandb:
            #     wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss,
            #                'val_acc': val_acc, 'sigma': sigma_list[epoch]}, step=epoch)
            #     if args.dynamic_noise:
            #         wandb.log({'accumulated_epsilon': accumulated_epsilon_list[epoch],
            #                    'accumulated_epsilon_bar': accumulated_epsilon_bar_list[epoch]}, step=epoch)


    load_checkpoint(checkpoint_path=checkpoint_name, net=net, optimizer=None)
    test_loss, test_acc = eval_model(net=net, train_loader=train_loader, eval_loader=val_loader, GP=GP)
    logger.info(f'Final test loss {test_loss:.2f} test accuracy {test_acc:.2f}')
    if args.wandb:
        wandb.log({'test_loss': test_loss, 'test_acc': test_acc})
        wandb.finish()


    return best_val_acc, checkpoint_name



