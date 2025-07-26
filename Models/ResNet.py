

import torch
import torch.nn as nn

from modules import MyFloor
import math

from torch.autograd import Function
from torch.nn import *

class dec_to_bin(Function):
    @staticmethod
    def forward(ctx, input, t):
        mask = 2 ** torch.arange(int(math.log2(t+1)) - 1, -1, -1).to(input.device)
        return input.int().unsqueeze(-1).bitwise_and(mask).ne(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.mean(dim=-1), None

convert_to_binary = dec_to_bin.apply

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

        self.relu = nn.ReLU(inplace=True)

    def hoyer_loss(self, x, t):
        x = convert_to_binary(x, t)
        if torch.sum(x) > 0:
            return torch.sum(x)
        return 0.0

    def forward(self, x, t, mode):
        x_initial = x
        for l in self.residual_function:
            x = l(x)
            if isinstance(l, MyFloor):
                #x = l(x)
                out_binary = x/l.up * t
                self.act_loss = self.hoyer_loss(out_binary.clone(), t)
                self.neuron_count = torch.numel(x)
            if 'ScaledNeuron' in str(l):
                self.act_loss = torch.count_nonzero(x)
                self.neuron_count = torch.numel(x)
        out = self.relu(x + self.shortcut(x_initial))
        out_binary = out/self.relu.up*t
        if mode=='ann':
            self.act_loss += self.hoyer_loss(out_binary.clone(), t)
        else:
            self.act_loss += torch.count_nonzero(out)
        self.neuron_count += torch.numel(out)
        return out

class BottleNeck(Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.residual_function(x) + self.shortcut(x))

class ResNet(Module):
    def __init__(self, block, num_block, num_classes=100):
        super().__init__()
        self.in_channels = 64
        self.conv1 = Sequential(
            Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True)
        )
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)


    def hoyer_loss(self, x, t):
        x = convert_to_binary(x, t)
        if torch.sum(x)>0:
            return torch.sum(x)

        return 0.0

    def forward(self, x, t, mode):
        out = x
        act_loss = 0
        self.neuron_count = 0
        ####conv1
        for l in self.conv1:
            out = l(out)
            if isinstance(l, MyFloor):
                act_loss += self.hoyer_loss((out/l.up)*t, t)
                self.neuron_count += torch.numel(out)
            if 'ScaledNeuron' in str(l):
                self.neuron_count += torch.numel(out)
                act_loss += torch.count_nonzero(out)


        for i, layers in enumerate([self.conv2_x, self.conv3_x, self.conv4_x, self.conv5_x]):
            for block in layers:
                out = block(out, t, mode)
                act_loss += block.act_loss
                self.neuron_count += block.neuron_count

        output = self.avg_pool(out)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output, act_loss

class ResNet4Cifar(Module):
    def __init__(self, block, num_block, num_classes=10):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Sequential(
            Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(16),
            ReLU(inplace=True)
        )
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 16, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 32, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 64, num_block[2], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

def resnet18(num_classes=10, **kargs):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet20(num_classes=10, **kargs):
    """ return a ResNet 20 object
    """
    return ResNet4Cifar(BasicBlock, [3, 3, 3], num_classes=num_classes)

def resnet34(num_classes=10, **kargs):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50(num_classes=10, **kargs):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)

def resnet101(num_classes=10, **kargs):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3],num_classes=num_classes)

def resnet152(num_classes=10, **kargs):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3],num_classes=num_classes)
