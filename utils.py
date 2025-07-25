import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import *
from modules import TCL, MyFloor, ScaledNeuron, StraightThrough

def isActivation(name):
    return (
        'relu' in name.lower() or
        'clip' in name.lower() or
        'floor' in name.lower() or
        'tcl' in name.lower()
    )

def replace_activation_by_module(model, m):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_module(module, m)
        if isActivation(module.__class__.__name__.lower()):
            if hasattr(module, "up"):
                print(module.up.item())
                model._modules[name] = m(module.up.item())
            else:
                model._modules[name] = m()
    return model

global num_features
def replace_activation_by_floor(model, t, threshold):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_floor(module, t, threshold)
        if isinstance(module, Conv2d):
            #print("Batch normalization")
            global num_features
            num_features = module.out_channels
        if isinstance(module, nn.Linear):
            #global num_features
            num_features = module.out_features
        if isActivation(module.__class__.__name__.lower()):
            if hasattr(module, "up"):
                print(module.up.item())
                if t == 0:
                    model._modules[name] = TCL()
                else:
                    #global num_features
                    model._modules[name] = MyFloor(module.up.item(), t, num_features, threshold)
                    #print(num_features)
            else:
                if t == 0:
                    model._modules[name] = TCL()
                else:
                    model._modules[name] = MyFloor(8., t, num_features, threshold)
    return model

def replace_activation_by_neuron(net):
    for name, mod in net._modules.items():
        cname = mod.__class__.__name__.lower()
        if hasattr(mod, "_modules"):
            net._modules[name] = replace_activation_by_neuron(mod)
        if isActivation(cname):
            print("yuP!", hasattr(mod, "up"))
            if hasattr(mod, "up"):
                net._modules[name] = ScaledNeuron(mod.up.item())
            else:
                net._modules[name] = ScaledNeuron(1.)
    return net

def replace_maxpool2d_by_avgpool2d(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_maxpool2d_by_avgpool2d(module)
        if module.__class__.__name__ == 'MaxPool2d':
            model._modules[name] = nn.AvgPool2d(kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding)
    return model

def reset_net(model):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            reset_net(module)
        if 'Neuron' in module.__class__.__name__:
            module.reset()
    return model

def _fold_bn(conv_module, bn_module, avg=False):
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)
    w_view = (conv_module.out_channels, 1, 1, 1)
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


def fold_bn_into_conv(conv_module, bn_module, avg=False):
    w, b = _fold_bn(conv_module, bn_module, avg)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b)
    else:
        conv_module.bias.data = b
    conv_module.weight.data = w
    # set bn running stats
    bn_module.running_mean = bn_module.bias.data
    bn_module.running_var = bn_module.weight.data ** 2

def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return (isinstance(m, nn.Conv2d)) or isinstance(m, nn.Linear)


def search_fold_and_remove_bn(model):
    model.eval()
    prev = None
    for n, m in model.named_children():
        if is_bn(m) and is_absorbing(prev):
            fold_bn_into_conv(prev, m)
            # set the bn module to straight through
            setattr(model, n, StraightThrough())
        elif is_absorbing(m):
            prev = m
        else:
            prev = search_fold_and_remove_bn(m)
    return prev


def regular_set(model, paras=([],[],[])):
    for n, module in model._modules.items():
        if isActivation(module.__class__.__name__.lower()) and hasattr(module, "up"):
            for name, para in module.named_parameters():
                paras[0].append(para)
        elif 'batchnorm' in module.__class__.__name__.lower():
            for name, para in module.named_parameters():
                paras[2].append(para)
        elif len(list(module.children())) > 0:
            paras = regular_set(module, paras)
        elif module.parameters() is not None:
            for name, para in module.named_parameters():
                paras[1].append(para)
    return paras
