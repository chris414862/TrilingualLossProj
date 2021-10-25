# Author: David Harwath

import math
import numpy as np
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_recalls(S):
    """
    Computes recall at 1, 5, and 10 given a similarity matrix S.
    By convention, rows of S are assumed to correspond to images and columns are captions.
    """
    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    if isinstance(S, torch.autograd.Variable):
        S = S.data
    n = S.size(0)
    A2I_scores, A2I_ind = S.topk(10, 0)
    I2A_scores, I2A_ind = S.topk(10, 1)
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    for i in range(n):
        A_foundind = -1
        I_foundind = -1
        for ind in range(10):
            if A2I_ind[ind, i] == i:
                I_foundind = ind
            if I2A_ind[i, ind] == i:
                A_foundind = ind
        # do r1s
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
        # do r5s
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)

    recalls = {'A_r1':A_r1.avg, 'A_r5':A_r5.avg, 'A_r10':A_r10.avg,
                'I_r1':I_r1.avg, 'I_r5':I_r5.avg, 'I_r10':I_r10.avg}
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls

def HowToNCE_loss(S, temperature):
    target = torch.zeros(S.size(0)).long().to(S.device)
    S = S.div(temperature)
    # Need to handle the first and last indices specially                                                                                                   
    score_tensors = []
    score_tensors.append(torch.cat([S[0,:], S[1:,0]]))
    for i in range(1, S.size(0) - 1):
        score_tensors.append(torch.cat([S[i,i].view(1), S[i, 0:i], S[i, i+1:], S[0:i, i], S[i+1:, i]]))
    score_tensors.append(torch.cat([S[-1,-1].view(1),S[-1,0:-1],S[0:-1,-1]]))
    loss = F.nll_loss(F.log_softmax(torch.stack(score_tensors), dim=1), target)
    return loss

def InfoNCE_loss(S, temperature):
    target = torch.LongTensor(list(range(S.size(0)))).to(S.device)
    S = S.div(temperature)
    I2C_loss = F.nll_loss(F.log_softmax(S, dim=1), target)
    #C2I_loss = F.nll_loss(F.log_softmax(S, dim=0), target) # original
    C2I_loss = F.nll_loss(F.log_softmax(S.t(), dim=1), target) # "fixed"
    loss = I2C_loss + C2I_loss
    return loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(base_lr, lr_ramp_steps, lr_decay, lr_decay_multiplier, optimizer, global_step):
    """Sets the learning rate to the initial LR decayed every lr_decay epochs"""
    if global_step < lr_ramp_steps:
        lr = base_lr * (global_step / lr_ramp_steps)
    else:
        lr = base_lr * (lr_decay_multiplier ** ((global_step - lr_ramp_steps) // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def load_progress(prog_pkl, quiet=False):
    """
    load progress pkl file
    Args:
        prog_pkl(str): path to progress pkl file
    Return:
        progress(list):
        epoch(int):
        global_step(int):
        best_epoch(int):
        best_avg_r10(float):
    """
    def _print(msg):
        if not quiet:
            print(msg)

    with open(prog_pkl, "rb") as f:
        prog = pickle.load(f)
        epoch, global_step, best_epoch, best_avg_r10, _ = prog[-1]

    _print("\nPrevious Progress:")
    msg =  "[%5s %7s %5s %7s %6s]" % ("epoch", "step", "best_epoch", "best_avg_r10", "time")
    _print(msg)
    return prog, epoch, global_step, best_epoch, best_avg_r10

def get_gpu_status_str():
    printlist = []
    for j in range(torch.cuda.device_count()):
        allocated = round(torch.cuda.memory_allocated(j)/1024**3,1)
        available = round(torch.cuda.get_device_properties(j).total_memory/1024**3,1)
        printlist.append('GPU%d %.1f/%.1f GB used' % (j, allocated, available))
    return ' '.join(printlist)
