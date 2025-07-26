import torch.multiprocessing as mp
import argparse

from Models import modelpool
from Preprocess import datapool
from argparse import ArgumentParser
from pathlib import Path
from funcs import *
from torch.nn import *
from utils import replace_activation_by_floor, replace_activation_by_neuron, replace_maxpool2d_by_avgpool2d


import torch.nn as nn
import os

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        'action', default='train', type=str, help='Action: train or test.'
    )
    parser.add_argument('--gpus', default=1, type=int, help='GPU number to use.')
    parser.add_argument('--bs', default=8, type=int, help='Batchsize')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--epochs', default=600, type=int, help='Training epochs')
    parser.add_argument('--id', default=None, type=str, help='Model identifier')
    parser.add_argument('--devices', default='0', type=str, help='gpu ids to use')
    parser.add_argument('--l', default=16, type=int, help='L')
    parser.add_argument('--t', default=16, type=int, help='T')
    parser.add_argument('--hoyer_decay', default=1e-8, type=float, help='hoyer decay')
    parser.add_argument('--mode', type=str, default='ann')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data', type=str, default='cifar100')
    parser.add_argument('--model', type=str, default='vgg16')
    parser.add_argument('--threshold', type=str, default='layer-wise', help='channel-wise or layer-wise: choice of MyFloor threshold')
    parser.add_argument('--is_pretrained', type=bool, default=False, help='want to use a pre-trained model or not')
    args = parser.parse_args()

    if args.devices == "cpu":
        dev = "cpu"
    else:
        dev = "cuda:0"
    act = args.action
    mode = args.mode

    path = Path("./saved_models")
    path.mkdir(parents = True, exist_ok=True)
    path = path / ("%s.pth" % args.id)

    print(act, mode)

    if args.gpus > 1:
        assert False
    else:
        train, l_te = datapool(args.data, args.bs)

        net = modelpool(args.model, args.data)
        net = replace_maxpool2d_by_avgpool2d(net)
        net = replace_activation_by_floor(net, t=args.l, threshold=args.threshold)
        crit = CrossEntropyLoss()
        if act == 'train':
            train_ann(
                train, l_te,
                net, args.epochs,
                dev,
                crit, args.l,
                args.hoyer_decay,
                args.lr, args.wd, args.id
            )
        elif act in {"test", "evaluate"}:
            net.load_state_dict(torch.load(path, map_location = dev))
            if mode == 'snn':
                net = replace_activation_by_neuron(net)
                print(net)
                net.to(dev)
                acc, sparsity = eval_snn(l_te, net, dev, args.t)
                print('Accuracy: ', acc)
                print('Sparsity: ', sparsity)
            elif mode == 'ann':
                net.to(dev)
                acc, _, sparsity = eval_ann(l_te, net, crit, dev, args.l, mode)
                print('Accuracy: {:.4f}'.format(acc))
                print('Sparsity: {:.4f}'.format(sparsity))
            else:
                AssertionError('Unrecognized mode')
