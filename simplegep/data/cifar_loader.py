import torch
import torchvision
from torch.utils.data import Subset
from torchvision import transforms


VAL_SIZE = 0.2


def get_transform_train():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.GaussianBlur(kernel_size=5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform_train


def get_transform_test():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform_test

def get_dataset(args):
    root = args.data_root

    train_dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=get_transform_train(),
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=get_transform_test(),
    )

    num_samples = len(train_dataset)
    num_val = int(num_samples * VAL_SIZE)
    num_train = num_samples - num_val

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(num_samples, generator=generator).tolist()

    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=get_transform_test())

    return train_subset, val_subset, test_set

def get_dataloaders(args):
    train_subset, val_subset, test_set = get_dataset(args)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batchsize, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args.batchsize, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batchsize, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader

def get_train_loader(root, batchsize: int):
    transform_train = get_transform_train()
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)
    return trainloader


def get_test_loader(root, batchsize: int):
    transform_test = get_transform_test()
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)
    return testloader


def get_num_samples(train=True) -> int:
    return 50_000 if train else 10_000
