import numpy as np
from torch import nn
import torch
from tqdm import tqdm
from utils import *
from modules import LabelSmoothing
import torch.distributed as dist
import random
import os

def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def eval_ann(test_dataloader, model, loss_fn, device, l=8, mode='ann'):
    epoch_loss = 0
    sparsity, neuron_count = 0, 0
    tot = torch.tensor(0.).cuda(device)
    model.to(device)
    model.eval()
    length = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            img = img.cuda(device)
            label = label.cuda(device)
            out, act_loss = model(img, l, mode)
            sparsity += act_loss
            neuron_count += model.neuron_count
            loss = loss_fn(out, label)
            epoch_loss += loss.item()
            length += len(label)
            tot += (label==out.max(1)[1]).sum().data
    return tot/length, epoch_loss/length, sparsity/neuron_count

def train_ann(
    l_tr,
    l_te,
    net,
    epochs,
    device, loss_fn, l, hoyer_decay,
    lr=0.1, wd=5e-4, save=None, parallel=False, rank=0
):
    print("NET", net, device)
    net.to(device)
    para1, para2, para3 = regular_set(net)
    optimizer = torch.optim.SGD([
                                {'params': para1, 'weight_decay': wd},
                                {'params': para2, 'weight_decay': wd},
                                {'params': para3, 'weight_decay': wd}
                                ],
                                lr=lr,
                                momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0
    for epoch in range(epochs):
        epoch_loss = 0
        length = 0
        sparsity = 0
        net.train()
        for img, label in l_tr:
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            out, act_loss = net(img, l)

            sparsity += act_loss
            loss = loss_fn(out, label) + hoyer_decay*act_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            length += len(label)
        tmp_acc, val_loss = eval_ann(l_te, net, loss_fn, device, l)# rank)
        if parallel:
            dist.all_reduce(tmp_acc)
        print('Epoch {} -> Val_loss: {}, Acc: {}'.format(epoch, val_loss, tmp_acc), flush=True)
        if rank == 0 and save != None and tmp_acc >= best_acc:
            torch.save(net.state_dict(), './saved_models/' + save + '.pth')
        best_acc = max(tmp_acc, best_acc)
        print('best_acc: ', best_acc)
        print('spiking_activity: ', sparsity)
        scheduler.step()


    return best_acc, net

def eval_snn(l_te, model, device, sim_len=8, mode='ann', rank=0, t=8):
    tot = torch.zeros(sim_len).cuda(device)
    length = 0
    sparsity, neuron_count = 0, 0
    model = model.cuda(device)
    model.eval()
    # valuate
    with torch.no_grad():
        for idx, (img, label) in enumerate(tqdm(l_te)):
            spikes = 0
            length += len(label)
            img = img.cuda(device)
            label = label.cuda(device)
            for t in range(sim_len):
                out, act_loss  = model(img, t, mode)
                sparsity += act_loss
                neuron_count += model.neuron_count
                spikes += out
                tot[t] += (label==spikes.max(1)[1]).sum()
            reset_net(model)
    neuron_count = neuron_count/(t+1)
    return tot/length, sparsity/neuron_count
