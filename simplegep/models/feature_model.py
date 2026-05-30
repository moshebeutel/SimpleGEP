import logging
from torch import nn
import torch.nn.functional as F

from simplegep.models.utils import initialize_weights, count_parameters


class DenseBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 use_batchnorm=False,
                 use_dropout=True,
                 activation='relu'):
        super(DenseBlock, self).__init__()

        self._fc = nn.Linear(in_channels, out_channels, bias=True)
        self._batch_norm = nn.GroupNorm(num_groups=out_channels//16, num_channels=out_channels, affine=True) if use_batchnorm \
            else nn.Identity(out_channels)
        self._act = nn.ReLU() if activation == 'relu' else nn.ELU()
        self._dropout = nn.Dropout(.5) if use_dropout else nn.Identity(out_channels)

    def forward(self, x):
        # return self._dropout(self._relu(self._batch_norm(self._fc(x))))
        return self._act(self._dropout(self._batch_norm(self._fc(x))))



class FeatureModel(nn.Module):
    def __init__(self, num_features=96, number_of_classes=26, depth_power=5,
                 cls_layer=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._output_info_fn = logging.info
        self._output_debug_fn = logging.debug
        self.cls_layer = cls_layer

        blocks = [DenseBlock((2**i) * num_features, (2**(i+1)) * num_features, use_batchnorm=True) for i in range(depth_power)]
        blocks.append(DenseBlock((2**depth_power) * num_features, (2**depth_power) * num_features, use_batchnorm=True))
        blocks.extend([DenseBlock((2**(i+1)) * num_features, (2**i) * num_features, use_batchnorm=True) for i in range(depth_power-1, -1, -1)])
        self._blocks = nn.ModuleList(blocks)
        self._extra_block = DenseBlock(num_features, int((1 / 2) * num_features),
                                        use_dropout=False, activation='relu', use_batchnorm=False)
        if self.cls_layer:
            self._output = nn.Linear(int((1 / 2) * num_features), number_of_classes)

        initialize_weights(self)

        self._output_info_fn(str(self))

        self._output_info_fn(f"Number Parameters: {count_parameters(self)}")

    def forward(self, x):
        self._output_debug_fn(f'input {x.shape}')

        for i, block in enumerate(self._blocks):
            x = block(x)
            self._output_debug_fn(f'output block {i} {x.shape}')

        x = self._extra_block(x)
        self._output_debug_fn(f'extra block {x.shape}')

        if self.cls_layer:
            logits = self._output(x)
            self._output_debug_fn(f'logits {logits.shape}')
            probs = F.softmax(logits, dim=1)
            self._output_debug_fn(f'softmax {probs.shape}')
            x = probs

        return x

