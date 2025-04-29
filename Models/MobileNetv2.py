"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import math
import torch
from modules import MyFloor
from torch.autograd import Function

__all__ = ['mobilenetv2']


class dec_to_bin(Function):
    @staticmethod
    def forward(ctx, input, t): 
        mask = 2 ** torch.arange(int(math.log2(t+1)) - 1, -1, -1).to(input.device)
        return input.int().unsqueeze(-1).bitwise_and(mask).ne(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.mean(dim=-1), None

convert_to_binary = dec_to_bin.apply

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def hoyer_loss(self, x, t):
        x = convert_to_binary(x, t)
        #self.save_output = x.clone()
        if torch.sum(x)>0: #  and l < self.start_spike_layer
            return torch.sum(x)
        
        return 0.0
    
    def forward(self, x, t):
        #if self.identity:
        #    return x + self.conv(x)
        #else:
        #    return self.conv(x)
        
        x_initial = x
        self.act_loss = 0
        self.neuron_count = 0
        for l in self.conv:
            x = l(x)
            if isinstance(l, MyFloor):
                #x = l(x)
                out_binary = x/l.up * t
                self.act_loss += self.hoyer_loss(out_binary.clone(), t)
                self.neuron_count = torch.numel(x)
            if 'ScaledNeuron' in str(l):
                self.act_loss = torch.count_nonzero(x)
                self.neuron_count = torch.numel(x)
        if self.identity:
            return x_initial + x
        else:
            return x

        out = self.relu(x + self.shortcut(x_initial))
        out_binary = out/self.relu.up*t
        self.act_loss += self.hoyer_loss(out_binary.clone(), t)
        return out


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        self.first_conv = conv_3x3_bn(3, input_channel, 2)
        layers = []
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def hoyer_loss(self, x, t):
        x = convert_to_binary(x, t)
        #self.save_output = x.clone()
        if torch.sum(x)>0: #  and l < self.start_spike_layer
            return torch.sum(x)
        
        return 0.0
    
    def forward(self, x, t, mode):

        out = x
        act_loss = 0
        self.neuron_count = 0
        ####conv1
        for l in self.first_conv:
            out = l(out)
            if isinstance(l, MyFloor):
                act_loss += self.hoyer_loss((out/l.up)*t, t)
                self.neuron_count += torch.numel(out)
            if 'ScaledNeuron' in str(l):
                act_loss += torch.count_nonzero(out)
                self.neuron_count += torch.numel(out)

        for i, layers in enumerate([self.features]):
            for block in layers:
                out = block(out, t)
                act_loss += block.act_loss
                self.neuron_count += block.neuron_count
                
        for l in self.conv:
            out = l(out)
            if isinstance(l, MyFloor):
                act_loss += self.hoyer_loss((out/l.up)*t, t)
                self.neuron_count += torch.numel(out)
            if 'ScaledNeuron' in str(l):
                act_loss += torch.count_nonzero(out)
                self.neuron_count += torch.numel(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out, act_loss

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)
