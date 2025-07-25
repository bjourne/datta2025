import torch.multiprocessing as mp
import argparse

from argparse import ArgumentParser
from Models import modelpool
from Preprocess import datapool
from funcs import *
from utils import replace_activation_by_floor, replace_activation_by_neuron, replace_maxpool2d_by_avgpool2d
from ImageNet.train import main_worker
import torch.nn as nn
import os

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('action', default='train', type=str, help='Action: train or test.')
    parser.add_argument('--gpus', default=1, type=int, help='GPU number to use.')
    parser.add_argument('--bs', default=8, type=int, help='Batchsize')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--epochs', default=600, type=int, help='Training epochs') # better if set to 300 for CIFAR dataset
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

    # only ImageNet using multiprocessing,
    if args.gpus > 1:
        if args.data.lower() != 'imagenet':
            AssertionError('Only ImageNet using multiprocessing.')
        mp.spawn(main_worker, nprocs=args.gpus, args=(args.gpus, args))
    else:
        train, test = datapool(args.data, args.bs)
        # preparing model
        model = modelpool(args.model, args.data)
        model = replace_maxpool2d_by_avgpool2d(model)
        model = replace_activation_by_floor(model, t=args.l, threshold=args.threshold)
        criterion = nn.CrossEntropyLoss()
        if args.action == 'train':
            train_ann(
                train, test,
                model, args.epochs,
                dev,
                criterion, args.l,
                args.hoyer_decay,
                args.lr, args.wd, args.id
            )
        elif args.action == 'test' or args.action == 'evaluate':
            model.load_state_dict(torch.load('./saved_models/' + args.id + '.pth', map_location= dev))
            if args.mode == 'snn':
                model = replace_activation_by_neuron(model)
                model.to('cuda:'+str(device_ids[0]))
                acc, sparsity = eval_snn(test, model, dev, args.t, args.mode)
                print('Accuracy: ', acc)
                print('Sparsity: ', sparsity)
            elif args.mode == 'ann':
                model.to('cuda:'+str(device_ids[0]))
                #model.cuda()
                acc, _, sparsity = eval_ann(test, model, criterion, dev, args.l, args.mode)
                print('Accuracy: {:.4f}'.format(acc))
                print('Sparsity: {:.4f}'.format(sparsity))
            else:
                AssertionError('Unrecognized mode')
