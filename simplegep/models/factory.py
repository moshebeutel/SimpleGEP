from simplegep.models.feature_model import FeatureModel
from simplegep.models.resnet import resnet20
from simplegep.models.tiny_cifar_net import tiny_cifar_net_4, tiny_cifar_net_8, tiny_cifar_net_16

model_hub = {'resnet20': resnet20, 'tiny_cifar_net_4': tiny_cifar_net_4,
             'tiny_cifar_net_8': tiny_cifar_net_8, 'tiny_cifar_net_16': tiny_cifar_net_16}

emg_datasets = ['putemg', 'keypressemg']


def get_model(args):
    if str.lower(args.dataset) in emg_datasets:
        return FeatureModel(num_features=args.num_features, number_of_classes=args.num_classes,
                         cls_layer=(not args.use_gp),
                         depth_power=args.depth_power)

    model_name = args.model_name
    if 'tiny' in model_name and 'filters' in args:
        model_name = f'tiny_cifar_net_{args.filter}'

    assert model_name in model_hub, 'Model {} not found'.format(model_name)
    return model_hub[model_name]()