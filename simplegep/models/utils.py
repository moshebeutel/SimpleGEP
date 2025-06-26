from torch import nn

def initialize_weights(module: nn.Module):
    for m in module.modules():

        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()

def count_parameters(model, return_layer_sizes=False):
    if not return_layer_sizes:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        layer_sizes = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                layer_sizes.append(param.numel())
        return sum(layer_sizes), layer_sizes
