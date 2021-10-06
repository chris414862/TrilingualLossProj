# Author: David Harwath

import math
import numpy as np
import pickle
import random
import torch
import torch.nn as nn
from collections import defaultdict



def calc_recalls(S, view1="", view2=""):
    """
    Computes recall at 1, 5, and 10 given a similarity matrix S.
    By convention, rows of S are assumed to correspond to images (view1) and columns are captions (view2).
    """
    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    if isinstance(S, torch.autograd.Variable):
        S = S.data
    n = S.size(0)
    v1_v2_scores, v1_v2_ind = S.topk(10, 1)#along the v2 dimension
    v2_v1_scores, v2_v1_ind = S.topk(10, 0)#along the v1 dimension
    v1_r1 = AverageMeter()
    v1_r5 = AverageMeter()
    v1_r10 = AverageMeter()
    v2_r1 = AverageMeter()
    v2_r5 = AverageMeter()
    v2_r10 = AverageMeter()
    for i in range(n):
        v1_foundind = -1
        v2_foundind = -1
        for ind in range(10):
            if v2_v1_ind[ind, i] == i:
                v1_foundind = ind
            if v1_v2_ind[i, ind] == i:
                v2_foundind = ind
        # do r1s
        if v2_foundind == 0:
            v2_r1.update(1)
        else:
            v2_r1.update(0)
        if v1_foundind == 0:
            v1_r1.update(1)
        else:
            v1_r1.update(0)
        # do r5s
        if v2_foundind >= 0 and v2_foundind < 5:
            v2_r5.update(1)
        else:
            v2_r5.update(0)
        if v1_foundind >= 0 and v1_foundind < 5:
            v1_r5.update(1)
        else:
            v1_r5.update(0)
        # do r10s
        if v2_foundind >= 0 and v2_foundind < 10:
            v2_r10.update(1)
        else:
            v2_r10.update(0)
        if v1_foundind >= 0 and v1_foundind < 10:
            v1_r10.update(1)
        else:
            v1_r10.update(0)

    recalls = defaultdict(dict)
    recalls.update({view2+"->"+view1:{'r1':v2_r1.avg, 'r5':v2_r5.avg, 'r10':v2_r10.avg},
           view1+"->"+view2:{'r1':v1_r1.avg, 'r5':v1_r5.avg, 'r10':v1_r10.avg}})
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls


class HypersphericLoss():
    def __init__(self, alpha=2, t=2, lam=1.):
        self.alpha=alpha
        self.t=t
        self.lam=lam

    def align_loss(self, view1, view2):
        return (view1 - view2).norm(p=2, dim=1).pow(self.alpha).mean()

    def u_single_view(self, view:torch.Tensor):
        sq_distances = torch.pdist(view, p=2).pow(2)
        return sq_distances.mul(-self.t).exp().mean().log()

    def uniformity_loss(self, view1, view2):
        return (self.u_single_view(view1) + self.u_single_view(view2))/2

    def __call__(self, view1, view2, **kwargs):
        al =self.align_loss(view1, view2) 
        ul = self.uniformity_loss(view1, view2)

        return al + self.lam*ul, {"al":al, "ul":ul}


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, use_hard_neg=False):
        self.margin = margin
        self.use_hard_neg = use_hard_neg


    def __call__(self, view1, view2, **kwargs):
        loss = 0.0
        aux_losses = None
        S = dot_sim_matrix(view1, view2)
        v1_to_v2_sampled_loss = sampled_triplet_loss_from_S(S, self.margin)
        v2_to_v1_sampled_loss = sampled_triplet_loss_from_S(S.t(), self.margin)
        unif_samp = v1_to_v2_sampled_loss + v2_to_v1_sampled_loss
        loss += unif_samp
        if self.use_hard_neg:
            aux_losses = dict()
            v1_to_v2_semihard_loss = semihardneg_triplet_loss_from_S(S, self.margin)
            v2_to_v1_semihard_loss = semihardneg_triplet_loss_from_S(S.t(), self.margin)
            hard_neg_samp = v1_to_v2_semihard_loss + v2_to_v1_semihard_loss
            loss += hard_neg_samp
            aux_losses["unif_samp"] = unif_samp
            aux_losses["hard_neg"] = hard_neg_samp

        return loss, aux_losses


def cosine_sim_matrix(view1, view2, epsilon=1e-8, debug=False):
    # view1, view2 dims: [batch,embed_size]
    view1_norm = torch.linalg.vector_norm(view1, ord=2, dim=1, keepdim=True)
    # view1_norm dims: [batch, 1]
    view2_norm = torch.linalg.vector_norm(view2, ord=2, dim=1, keepdim=True)
    # view2_norm dims: [batch, 1]
    
    dot_sim = torch.mm(view1, view2.transpose(0,1))
    # dot_sim dims: [batch,batch]
    norm_mat = torch.mm(view1_norm, view2_norm.transpose(0,1))
    # norm_mat dims: [batch, batch]

    norm_mat = torch.clamp(norm_mat, min=epsilon)
    # norm_mat dims: [batch, batch]

    cosine_sim = dot_sim / norm_mat
    # print("cosine_sim nans") if debug else ""
    # print(torch.isnan(cosine_sim).sum()) if debug else ""
    # cosine_sim dims: [batch, batch]

    return cosine_sim


def dot_sim_matrix(view1, view2, debug=False):
    S = torch.mm(view1, view2.t())
    return S
        

class MultiViewCodingLoss(nn.Module):
    def __init__(self, embed_size=None, sim_measure="cosine", batch_size=None, temperature=1.0):
        super().__init__()
        self.embed_size = embed_size
        self.similarity_measure = sim_measure
        self.epsilon = 1e-8 # from Pytorch implementation (I assume for numerical stability)
        self.tao = temperature #.1  from Mulit-view Coding paper
        self.batch_size = batch_size
        # This is from the infoNCE paper in the density ratio (approx.) function
        # they use k time steps, but we (might) need this to be a batch size
        self.num_bilinear_mats = 1 #self.batch_size

        if self.similarity_measure == "bilinear_approx":
            assert not batch_size is None, str(self.__class__)+": if similarity_measure=='{self.similarity_measure}" + \
                                           "batch_size needs to be given"
            self.bilin_mat = nn.Bilinear(self.embed_size, self.embed_size, self.num_bilinear_mats)
            self.den_ratio_func = self.bilinear_transform
        elif self.similarity_measure == "cosine":
            self.den_ratio_func = cosine_sim_matrix
        elif self.similarity_measure == "dot":
            self.den_ratio_func = dot_sim_matrix
        else:
            raise ValueError(str(self.__class__)+": 'similarity_measure' not recognized. "+
                            f"Received: {self.similarity_measure}")

    

    def bilinear_transform(self, view1, view2):
        """
        This is more similar to the implementation in the infoNCE paper
        """

        #TODO: Finish this function
        # view1, view2 dims: [batch,embed_size]
        den_ratios = self.bilin_mat(view1, view2)
        # den_ratios dims: [batch,embed_size]




    def forward(self, view1, view2, **kwargs):
        """
        Implementation of infoNCE loss.

        Parameters: 
            view1:   ( torch.Tensor ) 
                 Shape=[batch,embed_size]
            view2:   ( torch.Tensor ) 
                First Shape=[batch,embed_size]
            mask: ( torch.Tensor ) 
                Shape=[batch,embed_size]
        """
        density_ratios = self.den_ratio_func(view1, view2)#, debug=kwargs['debug'])
        density_ratios = density_ratios/self.tao
        # print("density_ratios", density_ratios, density_ratios.shape) if kwargs['debug'] else ""
        # print(density_ratios) if kwargs['debug'] else ""

        ## View 1 as anchor
        # print("density_ratios nans") if kwargs['debug'] else ""
        # print(torch.isnan(density_ratios).sum()) if kwargs['debug'] else ""
        log_sm_den_ratios = -torch.nn.functional.log_softmax(density_ratios, dim=-1) 
        # print("log_sm_density_ratios 1") if kwargs['debug'] else ""
        # print(log_sm_den_ratios) if kwargs['debug'] else ""
        correct = log_sm_den_ratios.diag()
        # print("correct 1") if kwargs['debug'] else ""
        # print(correct) if kwargs['debug'] else ""
        loss = torch.sum(correct)
        # print("loss 1") if kwargs['debug'] else ""
        # print(loss) if kwargs['debug'] else ""

        # batch scale
        loss = loss/correct.size(0)

        
        # ## View 2 as anchor
        log_sm_den_ratios = -torch.nn.functional.log_softmax(density_ratios.transpose(0,1), dim=-1) 
        correct = log_sm_den_ratios.diag()#torch.gather(log_sm_den_ratios, dim=0, index=indexer)
        loss2 = torch.sum(correct)

        # batch scale
        loss2 = loss2/(correct.size(0))
        # print("loss 2") if kwargs['debug'] else ""
        # print(loss2) if kwargs['debug'] else ""

        loss = loss + loss2
        auxillary_losses = None
        return loss, auxillary_losses

    



def one_imposter_index(i, N):
    imp_ind = random.randint(0, N - 2)
    if imp_ind == i:
        imp_ind = N - 1
    return imp_ind

def basic_get_imposter_indices(N):
    imposter_idc = []
    for i in range(N):
        # Select an imposter index for example i:
        imp_ind = one_imposter_index(i, N)
        imposter_idc.append(imp_ind)
    return imposter_idc

def semihardneg_triplet_loss_from_S(S, margin):
    """
    Input: Similarity matrix S
    Output: The one-way triplet loss from rows of S to columns of S. Impostors are taken
    to be the most similar point to the anchor that is still less similar to the anchor
    than the positive example.
    You would need to run this function twice, once with S and once with S.t(),
    in order to compute the triplet loss in both directions.
    """
    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    N = S.size(0)
    positive_scores = S.diag()
    mask = ((S - S.diag().view(-1,1)) < 0).float().detach()
    imposter_scores = (S * mask).max(dim=1).values
    loss = (imposter_scores - positive_scores + margin).clamp(min=0).mean()
    return loss

def sampled_triplet_loss_from_S(S, margin):
    """
    Input: Similarity matrix S
    Output: The one-way triplet loss from rows of S to columns of S. Imposters are
    randomly sampled from the columns of S.
    You would need to run this function twice, once with S and once with S.t(),
    in order to compute the triplet loss in both directions.
    """
    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    N = S.size(0)
    positive_scores = S.diag()
    imp_indices = np.random.randint(0, N-1, size=N)
    for j, ind in enumerate(imp_indices):
        if ind >= j:
            imp_indices[j] = ind + 1
    imposter_scores = S[range(N), imp_indices]
    loss = (imposter_scores - positive_scores + margin).clamp(min=0).mean()
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

def adjust_learning_rate(base_lr, lr_decay, lr_decay_multiplier, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed every lr_decay epochs"""
    lr = base_lr * (lr_decay_multiplier ** (epoch // lr_decay))
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


def computeMatchmap(I, A):
    assert(I.dim() == 3)
    assert(A.dim() == 2)
    D = I.size(0)
    H = I.size(1)
    W = I.size(2)
    T = A.size(1)                                                                                                                     
    Ir = I.view(D, -1).t()
    matchmap = torch.mm(Ir, A)
    matchmap = matchmap.view(H, W, T)  
    return matchmap

def matchmapSim(M, simtype):
    assert(M.dim() == 3)
    if simtype == 'SISA':
        return M.mean()
    elif simtype == 'MISA':
        M_maxH, _ = M.max(0)
        M_maxHW, _ = M_maxH.max(0)
        return M_maxHW.mean()
    elif simtype == 'SIMA':
        M_maxT, _ = M.max(2)
        return M_maxT.mean()
    else:
        raise ValueError

def sampled_margin_rank_loss(image_outputs, audio_outputs, nframes, margin=1., simtype='MISA'):
    """
    Computes the triplet margin ranking loss for each anchor image/caption pair
    The impostor image/caption is randomly sampled from the minibatch
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    for i in range(n):
        I_imp_ind = i
        A_imp_ind = i
        while I_imp_ind == i:
            I_imp_ind = np.random.randint(0, n)
        while A_imp_ind == i:
            A_imp_ind = np.random.randint(0, n)
        nF = nframes[i]
        nFimp = nframes[A_imp_ind]
        anchorsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[i][:, 0:nF]), simtype)
        Iimpsim = matchmapSim(computeMatchmap(image_outputs[I_imp_ind], audio_outputs[i][:, 0:nF]), simtype)
        Aimpsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[A_imp_ind][:, 0:nFimp]), simtype)
        A2I_simdif = margin + Iimpsim - anchorsim
        if (A2I_simdif.data > 0).all():
            loss = loss + A2I_simdif
        I2A_simdif = margin + Aimpsim - anchorsim
        if (I2A_simdif.data > 0).all():
            loss = loss + I2A_simdif
    loss = loss / n
    return loss

def compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype='MISA'):
    """
    Assumes image_outputs is a (batchsize, embedding_dim, rows, height) tensor
    Assumes audio_outputs is a (batchsize, embedding_dim, 1, time) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    S = torch.zeros(n, n, device=image_outputs.device)
    for image_idx in range(n):
            for audio_idx in range(n):
                nF = max(1, nframes[audio_idx])
                S[image_idx, audio_idx] = matchmapSim(computeMatchmap(image_outputs[image_idx], audio_outputs[audio_idx][:, 0:nF]), simtype)
    return S

def compute_pooldot_similarity_matrix(image_outputs, audio_outputs, nframes):
    """
    Assumes image_outputs is a (batchsize, embedding_dim, rows, height) tensor
    Assumes audio_outputs is a (batchsize, embedding_dim, 1, time) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    S[i][j] is computed as the dot product between the meanpooled embeddings of
    the ith image output and jth audio output
    """
    # assert(image_outputs.dim() == 4)
    # assert(audio_outputs.dim() == 4)
    # n = image_outputs.size(0)
    # imagePoolfunc = nn.AdaptiveAvgPool2d((1, 1))
    # pooled_image_outputs = imagePoolfunc(image_outputs).squeeze(3).squeeze(2)
    # audioPoolfunc = nn.AdaptiveAvgPool2d((1, 1))
    # pooled_audio_outputs_list = []
    # for idx in range(n):
    #     # nF = max(1, nframes[idx])
    #     pooled_audio_outputs_list.append(audio_outputs[idx][:, :, 0:nF])#audioPoolfunc(audio_outputs[idx][:, :, 0:nF]).unsqueeze(0))
    # pooled_audio_outputs = torch.cat(pooled_audio_outputs_list).squeeze(3).squeeze(2)
    S = torch.mm(image_outputs, audio_outputs.t())
    return S


