from torch import nn

class TinyCifarNet(nn.Module):
    def __init__(self, num_filters = 4):
        super(TinyCifarNet, self).__init__()
        self.representation_size = num_filters * 7 * 7
        self.conv1 = nn.Conv2d(3, num_filters, 3, padding=1, stride=2)
        # self.bn1 = nn.BatchNorm2d(num_filters, affine=False)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(3, stride=2)
        self.fc1 = nn.Linear(self.representation_size, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, self.representation_size)
        x = self.fc1(x)
        assert x.size(0) == batch_size, f'output shape error. Expected {batch_size} but got {x.size(0)}'
        return x

def tiny_cifar_net_4():
    return TinyCifarNet(num_filters=4)

def tiny_cifar_net_8():
    return TinyCifarNet(num_filters=8)

def tiny_cifar_net_16():
    return TinyCifarNet(num_filters=16)