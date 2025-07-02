import gc
import logging
import torch
import torchvision
from tqdm import tqdm
from simplegep.data.cifar_loader import get_train_loader, get_test_loader
from simplegep.dp.dp_params import get_dp_params
from simplegep.dp.dp_sgd import GradsProcessor
from simplegep.dp.per_sample_grad import pretrain_actions, backward_pass_get_batch_grads, PublicDataPerSampleGrad
from simplegep.embeddings.embedder import Embedder
from simplegep.embeddings.svd_embedder import SVDEmbedder
from simplegep.models.factory import get_model
from simplegep.models.utils import initialize_weights, count_parameters
from simplegep.trainers.loss_function_factory import get_loss_function
from simplegep.trainers.optimizer_factory import get_optimizer
from simplegep.utils import eval_model


def train_epoch(net, loss_function, optimizer, train_loader, grads_processor, embedder: Embedder, pub_data_grads: PublicDataPerSampleGrad):
    pub_grads = pub_data_grads.get_grads(current_state_dict=net.state_dict())
    embedder.calc_embedding_space(pub_grads)


    train_loss, train_acc = 0.0, 0.0
    net.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (data, target) in pbar:
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = net(data)
        loss = loss_function(output, target)
        correct_predictions = torch.eq(output.argmax(dim=1), target)
        batch_acc = correct_predictions.sum().item() / len(correct_predictions)
        train_acc += batch_acc
        train_loss += loss.item()
        flat_per_sample_grads = backward_pass_get_batch_grads(batch_loss=loss, net=net)
        processed_grads = grads_processor.process_grads(flat_per_sample_grads)
        offset = 0
        for param in net.parameters() :
            numel = param.numel()
            grad = processed_grads[offset:offset+numel].reshape(param.shape)
            param.grad = grad.clone().reshape(param.shape)
            offset += numel
        optimizer.step()
        pbar.set_description(f'Batch {batch_idx}/{len(train_loader)} train loss {loss.item()} train accuracy {batch_acc}')

        data, target, output, loss, correct_predictions = data.detach().cpu(), target.detach().cpu(), output.detach().cpu(), loss.detach().cpu(), correct_predictions.detach().cpu()
        data, target, output, loss, correct_predictions = None, None, None, None, None
        del data, target, output, loss, correct_predictions
        gc.collect()
        torch.cuda.empty_cache()
    return train_loss, train_acc


def train(args, logger: logging.Logger):
    logger.info('Starting training')

    net = get_model(args.model_name)
    initialize_weights(net)
    num_params, layer_sizes = count_parameters(model=net, return_layer_sizes=True)
    logger.debug(f'Model set to {args.model_name} num params {num_params}')
    logger.debug(f'layer sizes: {layer_sizes}')

    reduction = 'sum' if args.private else 'mean'
    loss_function = get_loss_function(args.loss_function, reduction=reduction)
    logger.debug(f'loss function set to {args.loss_function} reduction {reduction}')

    net, loss_function = pretrain_actions(model=net, loss_func=loss_function)
    logger.debug('model and loss functions prepared for per sample grads')

    optimizer = get_optimizer(optimizer_name=args.optimizer, model=net, lr=args.lr)
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
                 f' epsilon {dp_params.epsilon}')

    grads_processor = GradsProcessor(clip_strategy_name=args.clip_strategy,
                                     noise_multiplier=dp_params.sigma,
                                     clip_value=args.clip_value)

    logger.debug(f'Created GradsProcessor with strategy {args.clip_strategy} '
                 f'noise multiplier {dp_params.sigma}'
                 f' clip value {args.clip_value}')

    embedder = SVDEmbedder(args.num_basis)

    logger.debug(f'Created SVDEmbedder with {args.num_basis} basis elements')

    public_inputs, public_targets = get_aux_data(aux_dataset=args.aux_dataset,
                                                 aux_data_size=args.aux_data_size,
                                                 real_labels=args.real_labels)

    logger.debug(f'Created public data with {len(public_inputs)} examples')

    pub_data_grads = PublicDataPerSampleGrad(public_data=(public_inputs, public_targets), net=net)

    logger.debug(f'Created PublicDataPerSampleGrad')

    net = net.cuda()
    for epoch in range(args.num_epochs):
        logger.info(f'***** Starting epoch {epoch}  ******')
        train_loss, train_acc = train_epoch(net=net, loss_function=loss_function, optimizer=optimizer,
                                            train_loader=train_loader, grads_processor=grads_processor,
                                            embedder=embedder, pub_data_grads=pub_data_grads)
        logger.info(
            f'Epoch {epoch}/{args.num_epochs} train loss {train_loss} train accuracy {train_acc}')
        test_loss, test_acc = eval_model(net=net, loss_function=loss_function, loader=test_loader)
        logger.info(f'Epoch {epoch}/{args.num_epochs} test loss {test_loss} test accuracy {test_acc}')


def get_aux_data(aux_dataset: str, aux_data_size: int, real_labels: bool):
    ## preparing auxiliary data
    num_public_examples = aux_data_size
    if ('cifar' in aux_dataset):
        if (aux_dataset == 'cifar100'):
            transform_test = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        public_data_loader = torch.utils.data.DataLoader(testset, batch_size=num_public_examples, shuffle=False,
                                                         num_workers=2)  #
        for public_inputs, public_targets in public_data_loader:
            break
    else:
        public_inputs = torch.load(
            './data/imagenet_examples_2000')[
                        :num_public_examples]
    if (not real_labels):
        public_targets = torch.randint(high=10, size=(num_public_examples,))
    return public_inputs, public_targets




