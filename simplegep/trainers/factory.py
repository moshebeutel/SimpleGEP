
import torch

from simplegep.dp.per_sample_grad import PublicDataPerSampleGradProvider

loss_function_hub = {'cross_entropy': torch.nn.CrossEntropyLoss}


def get_loss_function(loss_function_name, reduction):
    assert loss_function_name in loss_function_hub, 'Loss function not found'
    loss_function_ctor = loss_function_hub[loss_function_name]
    loss_function = loss_function_ctor(reduction=reduction)
    return loss_function


optimizer_hub = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam}


def get_optimizer(args, model):
    assert args.optimizer in optimizer_hub, 'Optimizer not found'
    optimizer_func = optimizer_hub[args.optimizer]
    param_keys = ['lr', 'weight_decay'] if args.optimizer == 'adam' else ['lr', 'weight_decay', 'momentum']
    param_dict = {key:args.__dict__[key] for key in param_keys}
    optimizer = optimizer_func(model.parameters(), **param_dict)
    return optimizer


def get_dataloaders(args):
    if args.dataset == 'cifar10':
        from simplegep.data.cifar_loader import  get_dataloaders
    elif args.dataset == 'putemg':
        from simplegep.data.putemg_loader import get_dataloaders
    elif args.dataset == 'keypressemg':
        from simplegep.data.keypressemg_loader import get_dataloaders
    else:
        raise ValueError(f'Dataset {args.dataset} not supported')
    return get_dataloaders(args)

def get_public_grads_provider(args, net):
    import pathlib
    num_public_examples = args.aux_data_size
    aux_data_root = pathlib.Path(args.aux_data_root)
    assert aux_data_root.exists(), 'Auxiliary data root not found'
    assert aux_data_root.is_dir(), 'Auxiliary data root is not a directory'
    if args.aux_dataset == 'imagenet':
        public_inputs = torch.load(
            aux_data_root / 'imagenet_examples_2000')[:num_public_examples]
        assert not args.real_labels, 'Expected use of random labels'
        public_targets = torch.randint(high=args.num_classes, size=(num_public_examples,))
        pub_data_grads_provider = PublicDataPerSampleGradProvider(public_data=(public_inputs, public_targets), net=net,
                                                                  public_batchsize=args.batchsize)
    elif args.aux_dataset in ['keypressemg', 'putemg']:
        if args.aux_dataset == 'keypressemg':
            from simplegep.data.keypressemg_loader import get_dataloaders
        else:
            from simplegep.data.putemg_loader import get_dataloaders

        import copy

        aux_args = copy.copy(args)
        aux_args.dataset = args.aux_dataset
        aux_args.data_root = args.aux_data_root
        aux_data_loader, _, _ = get_dataloaders(aux_args)

        pub_data_grads_provider = PublicDataPerSampleGradProvider(public_data=aux_data_loader, net=net,
                                                                  public_batchsize=args.batchsize)
    else:
        raise ValueError(f'Dataset {args.aux_dataset} not supported')



    return pub_data_grads_provider

if __name__ == '__main__':
    import argparse
    import pathlib
    parser = argparse.ArgumentParser()
    project_root = pathlib.Path(__file__).parent.parent
    assert project_root.exists(), 'Project root not found'
    assert project_root.name == 'simplegep', 'Project root not found'
    dataset_root = project_root / 'data/CIFAR10'
    parser.add_argument('--aux_dataset', default='imagenet', type=str,)
    parser.add_argument('--aux_data_root', default=dataset_root.as_posix(), type=str,)
    parser.add_argument('--aux_data_size', default=2000, type=int,)
    args = parser.parse_args()
    provider = get_public_grads_provider(args)