import gc
import logging
from pathlib import Path
import torch
import torchvision
import wandb
from tqdm import tqdm
import simplegep.embeddings.factory
from simplegep import embeddings
from simplegep.data.cifar_loader import get_train_loader, get_test_loader
from simplegep.dp.dp_params import get_dp_params
from simplegep.dp.dp_sgd import GradsProcessor
from simplegep.dp.dynamic_dp import get_varying_sigma_values, get_decrease_function
from simplegep.dp.per_sample_grad import pretrain_actions, backward_pass_get_batch_grads, PublicDataPerSampleGrad
from simplegep.embeddings.embedder import Embedder
from simplegep.models.factory import get_model
from simplegep.models.utils import initialize_weights, count_parameters
from simplegep.trainers.loss_function_factory import get_loss_function
from simplegep.trainers.optimizer_factory import get_optimizer
from simplegep.utils import eval_model


def train_epoch(net, loss_function, optimizer, train_loader, grads_processor, embedder: Embedder,
                pub_data_grads: PublicDataPerSampleGrad):
    pub_grads = pub_data_grads.get_grads(current_state_dict=net.state_dict())
    embedder.calc_embedding_space(pub_grads)
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

        # get per sample grads
        flat_per_sample_grads = backward_pass_get_batch_grads(batch_loss=loss, net=net)

        # perturb grads in lower dimension subspace
        embedded_grads = embedder.embed(flat_per_sample_grads)
        processed_embeddings = grads_processor.process_grads(embedded_grads)
        reconstructed_grads = embedder.project_back(processed_embeddings)

        # substitute perturbed grads
        processed_grads = reconstructed_grads.squeeze()
        offset = 0
        for param in net.parameters():
            numel = param.numel()
            grad = processed_grads[offset:offset + numel].reshape(param.shape).to(param.device)
            param.grad = grad.clone().reshape(param.shape)
            offset += numel

        # update net parameters
        optimizer.step()

        pbar.set_description(
            f'Batch {batch_idx}/{len(train_loader)} train loss {loss.item()} train accuracy {batch_acc}')

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

    dp_params = get_dp_params(batchsize=args.batchsize,
                              num_training_samples=len(train_loader.dataset),
                              num_epochs=args.num_epochs,
                              epsilon=args.eps)
    logger.debug(
        f'DP params set: '
        f' batchsize {args.batchsize}'
        f' num_training_samples {len(train_loader.dataset)}'
        f' num_epochs {args.num_epochs}'
        f' epsilon {args.eps}'
    )
    logger.debug(f'DP params - '
                 f' sigma {dp_params.sigma}'
                 f' delta {dp_params.delta} '
                 f' epsilon {dp_params.epsilon}'
                 f' sampling prob {dp_params.sampling_prob} '
                 f' steps {dp_params.steps} ')

    sigma_list = [dp_params.sigma] * args.num_epochs
    if args.dynamic_noise:
        sigma_decrease_function = get_decrease_function(args)
        logger.debug(f'Using decrease function {sigma_decrease_function.__name__}')
        sigma_list, accumulated_epsilon_list, accumulated_epsilon_bar_list, sigma_orig = (
            get_varying_sigma_values(q=dp_params.sampling_prob,
                                     n_epoch=args.num_epochs,
                                     eps=args.eps, delta=dp_params.delta,
                                     initial_sigma_factor=args.dynamic_noise_high_factor,
                                     final_sigma_factor=args.dynamic_noise_low_factor,
                                     decrease_func=sigma_decrease_function))

        logger.info(f'Created varying sigma list with {len(sigma_list)} values')
        logger.debug(f'Sigma list: {sigma_list}')
        logger.debug(f'Accumulated epsilon list: {accumulated_epsilon_list}')
        logger.debug(f'Accumulated epsilon bar list: {accumulated_epsilon_bar_list}')
        logger.debug(f'Sigma orig: {sigma_orig}')

    grads_processor = GradsProcessor(clip_strategy_name=args.clip_strategy,
                                     noise_multiplier=dp_params.sigma,
                                     clip_value=args.clip_value)

    logger.debug(f'Created GradsProcessor with strategy {args.clip_strategy} '
                 f'noise multiplier {dp_params.sigma}'
                 f' clip value {args.clip_value}')

    num_epochs = min(args.num_epochs, len(sigma_list)) if args.dynamic_noise else args.num_epochs

    embedder = embeddings.factory.get_embedder(args)

    logger.debug(f'Created {args.embedder} embedder with {args.num_basis} basis elements')

    public_inputs, public_targets = get_aux_data(aux_data_root=args.data_root,
                                                 aux_dataset=args.aux_dataset,
                                                 aux_data_size=args.aux_data_size,
                                                 real_labels=args.real_labels)

    logger.debug(f'Created public data with {len(public_inputs)} examples')

    pub_data_grads = PublicDataPerSampleGrad(public_data=(public_inputs, public_targets), net=net)

    logger.debug(f'Created PublicDataPerSampleGrad')

    net = net.cuda()
    for epoch in range(num_epochs):
        logger.info(f'***** Starting epoch {epoch}  ******')
        train_loss, train_acc = train_epoch(net=net, loss_function=loss_function, optimizer=optimizer,
                                            train_loader=train_loader, grads_processor=grads_processor,
                                            embedder=embedder, pub_data_grads=pub_data_grads)
        logger.info(f'Epoch {epoch}/{args.num_epochs} train loss {train_loss:.2f} train accuracy {train_acc:.2f}')
        test_loss, test_acc = eval_model(net=net, loss_function=loss_function, loader=test_loader)
        logger.info(f'Epoch {epoch}/{args.num_epochs} test loss {test_loss:.2f} test accuracy {test_acc:.2f}')
        if args.wandb:
            wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'test_loss': test_loss,
                       'test_acc': test_acc, 'sigma': sigma_list[epoch]}, step=epoch)
            if args.dynamic_noise:
                wandb.log({'accumulated_epsilon': accumulated_epsilon_list[epoch],
                           'accumulated_epsilon_bar': accumulated_epsilon_bar_list[epoch]}, step=epoch)


def get_aux_data(aux_data_root: Path, aux_dataset: str, aux_data_size: int, real_labels: bool):
    ## preparing auxiliary data
    num_public_examples = aux_data_size
    if ('cifar' in aux_dataset):
        if (aux_dataset == 'cifar100'):
            transform_test = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            testset = torchvision.datasets.CIFAR100(root=aux_data_root, train=False, download=True,
                                                    transform=transform_test)
        public_data_loader = torch.utils.data.DataLoader(testset, batch_size=num_public_examples, shuffle=False,
                                                         num_workers=2)  #
        for public_inputs, public_targets in public_data_loader:
            break
    else:
        public_inputs = torch.load(
            aux_data_root / 'imagenet_examples_2000')[:num_public_examples]
    if (not real_labels):
        public_targets = torch.randint(high=10, size=(num_public_examples,))
    return public_inputs, public_targets
