import torch.distributed as dist
import random
import os


import numpy as np
import torch

from itertools import islice
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from utils import *

def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def eval_ann(l_te, net, loss_fn, device, l=8, mode='ann'):
    epoch_loss = 0
    sparsity, neuron_count = 0, 0
    tot = torch.tensor(0.).to(device)
    net.to(device)
    net.eval()
    length = 0
    with torch.no_grad():
        for img, label in l_te:
            img = img.to(device)
            label = label.to(device)
            out, act_loss = net(img, l, mode)
            sparsity += act_loss
            neuron_count += net.neuron_count
            loss = loss_fn(out, label)
            epoch_loss += loss.item()
            length += len(label)
            tot += (label==out.max(1)[1]).sum().data
    return tot/length, epoch_loss/length, sparsity/neuron_count

def train_ann(
    l_tr, l_te,
    net, epochs,
    dev, loss_fn, l, hoyer_decay,
    lr=0.1, wd=5e-4, save=None, parallel=False, rank=0
):
    print("NET", net, dev)
    net.to(dev)
    para1, para2, para3 = regular_set(net)
    opt_groups = [
        {'params': para1, 'weight_decay': wd},
        {'params': para2, 'weight_decay': wd},
        {'params': para3, 'weight_decay': wd}
    ]
    optimizer = SGD(opt_groups, lr=lr, momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0
    for epoch in range(epochs):
        epoch_loss = 0
        length = 0
        sparsity = 0
        net.train()
        for xs, label in islice(l_tr, 20):
            xs = xs.to(dev)
            label = label.to(dev)
            optimizer.zero_grad()
            out, act_loss = net(xs, l, "ann")

            sparsity += act_loss
            loss = loss_fn(out, label) + hoyer_decay*act_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            length += len(label)
        tmp_acc, val_loss, _ = eval_ann(l_te, net, loss_fn, dev, l)
        if parallel:
            dist.all_reduce(tmp_acc)
        fmt = 'Epoch {} -> Val_loss: {}, Acc: {}'
        print(fmt.format(epoch, val_loss, tmp_acc))
        if rank == 0 and save != None and tmp_acc >= best_acc:
            torch.save(net.state_dict(), './saved_models/' + save + '.pth')
        best_acc = max(tmp_acc, best_acc)
        print('best_acc: ', best_acc)
        print('spiking_activity: ', sparsity)
        scheduler.step()

    return best_acc, net

def eval_snn(l_te, model, dev, sim_len, mode, t=8):
    tot = torch.zeros(sim_len).to(dev)
    length = 0
    sparsity, neuron_count = 0, 0
    model = model.to(dev)
    model.eval()
    # valuate
    with torch.no_grad():
        for idx, (xs, ys) in enumerate(tqdm(l_te)):
            spikes = 0
            length += len(ys)
            xs = xs.to(dev)
            ys = ys.to(dev)
            for t in range(sim_len):
                out, act_loss = model(xs, t, mode)
                sparsity += act_loss
                neuron_count += model.neuron_count
                spikes += out
                tot[t] += (ys==spikes.max(1)[1]).sum()
            reset_net(model)
    # This is wrong
    neuron_count = neuron_count/(t+1)
    return tot / length, sparsity / neuron_count
