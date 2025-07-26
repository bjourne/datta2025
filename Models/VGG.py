from unicodedata import numeric
import torch.nn as nn
import torch
from modules import MyFloor
import math

from spikingjelly.clock_driven import neuron
from torch.autograd import Function
from torch.nn import *

class dec_to_bin(Function):
    @staticmethod
    def forward(ctx, input, t):
        mask = 2 ** torch.arange(int(math.log2(t+1))).to(input.device)
        return (input.int().unsqueeze(-1).bitwise_and(mask).ne(0).float()).clone().detach()

    @staticmethod
    def backward(ctx, grad_output):
        #return grad_output.mean(dim=-1), None
        return (grad_output.mean(dim=-1)).clone().detach(), None

convert_to_binary = dec_to_bin.apply

cfg = {
    'VGG11': [
        [64, 'M'],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG13': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ]
}


class VGG(Module):
    def __init__(self, vgg_name, num_classes, dropout):
        super(VGG, self).__init__()
        self.init_channels = 3
        self.num_classes = num_classes
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout)
        self.neuron = neuron.IFNode(v_reset=None)
        if num_classes == 1000:
            self.classifier = Sequential(
                Flatten(),
                Linear(512*7*7, 4096),
                ReLU(inplace=True),
                Dropout(dropout),
                Linear(4096, 4096),
                ReLU(inplace=True),
                Dropout(dropout),
                Linear(4096, num_classes)
            )
        else:
            self.classifier = Sequential(
                Flatten(),
                Linear(512, 4096),
                ReLU(inplace=True),
                Dropout(dropout),
                Linear(4096, 4096),
                ReLU(inplace=True),
                Dropout(dropout),
                Linear(4096, num_classes)
            )

        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, Linear):
                nn.init.zeros_(m.bias)

    def _make_layers(self, cfg, dropout):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(BatchNorm2d(x))
                layers.append(ReLU(inplace=True))
                layers.append(Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers)

    def hoyer_loss(self, x, t):
        x = convert_to_binary(x, t)

        if torch.sum(x)>0:
            return torch.sum(x)
        return 0.0

    def forward(self, x, t, mode):
        act_loss = 0.0
        self.neuron_count = 0
        out = x
        batch_size, _, _, _ = x.shape
        counter = 0
        time_steps = int(math.log2(t+1))

        for i, layers in enumerate([
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4,
                self.layer5,
                self.classifier
        ]):
            for l in layers:
                out = l(out)

                if isinstance(l, MyFloor):
                    self.neuron_count += torch.numel(out)
                    act_loss += self.hoyer_loss((out/l.up)*t, t)
                if 'ScaledNeuron' in str(l):
                    self.neuron_count += torch.numel(out)
                    act_loss += torch.count_nonzero(out)

        return (out), act_loss


class VGG_normed(nn.Module):
    def __init__(self, vgg_name, num_classes, dropout):
        super(VGG_normed, self).__init__()
        self.num_classes = num_classes
        self.module_list = self._make_layers(cfg[vgg_name], dropout)


    def _make_layers(self, cfg, dropout):
        layers = []
        for i in range(5):
            for x in cfg[i]:
                if x == 'M':
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    layers.append(nn.Conv2d(3, x, kernel_size=3, padding=1))
                    layers.append(nn.ReLU(inplace=True))
                    layers.append(nn.Dropout(dropout))
                    self.init_channels = x
        layers.append(nn.Flatten())
        if self.num_classes == 1000:
            layers.append(nn.Linear(512*7*7, 4096))
        else:
            layers.append(nn.Linear(512, 4096))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(4096, 4096))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(4096, self.num_classes))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.module_list(x)



def vgg11(num_classes=10, dropout=0, **kargs):
    return VGG('VGG11', num_classes, dropout)


def vgg13(num_classes=10, dropout=0, **kargs):
    return VGG('VGG13', num_classes, dropout)


def vgg16(num_classes=10, dropout=0, **kargs):
    return VGG('VGG16', num_classes, dropout)


def vgg19(num_classes=10, dropout=0, **kargs):
    return VGG('VGG19', num_classes, dropout)


def vgg16_normed(num_classes=10, dropout=0, **kargs):
    return VGG_normed('VGG16', num_classes, dropout)
