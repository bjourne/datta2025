from torch import nn
import torch.nn.functional as F
import numpy as np
import torch

from torch.autograd import Function
from torch.nn import Module
from spikingjelly.clock_driven.neuron import IFNode


class StraightThrough(Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input

class ScaledNeuron(Module):
    def __init__(self, scale=1.):
        super(ScaledNeuron, self).__init__()
        self.scale = scale
        self.t = 0
        self.neuron = IFNode(v_reset=None)

    def forward(self, x):
        x = x / self.scale
        if self.t == 0:
            self.neuron(torch.ones_like(x)*0.5, 1.0, True)
        x = self.neuron(x, 1.0, True)
        self.t += 1
        return x * self.scale

    def reset(self):
        self.t = 0
        self.neuron.reset()

class GradFloor(Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

myfloor = GradFloor.apply

class MyFloor(nn.Module):
    def __init__(self, up=8., t=32, channels=1, threshold='channel-wise'): #channels=64
        super().__init__()

        self.channels = channels
        thres = torch.ones(self.channels)*up
        if self.channels<4000:
            thres = thres.view(1,self.channels,1,1)
        else:
            thres = thres.view(1,self.channels)
        if (threshold == 'channel-wise'):
            self.up = nn.Parameter(thres, requires_grad=True)
        else:
            self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.t = t


    def forward(self, x):
        #if self.channels > 500:
        #    print(x.shape)
        x = x / self.up
        #if self.channels > 500:
        #    print(x.shape)
        x = myfloor(x*self.t+0.5)/self.t
        x = torch.clamp(x, 0, 1)
        x = x * self.up
        #print(x.shape)
        return x

class TCL(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Parameter(torch.Tensor([4.]), requires_grad=True)
    def forward(self, x):
        x = F.relu(x, inplace='True')
        x = self.up - x
        x = F.relu(x, inplace='True')
        x = self.up - x
        return x
