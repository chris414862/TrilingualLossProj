import torch
import torch.nn as nn
import numpy as np

from .loss_func_utils import dot_sim_matrix, cosine_sim_matrix


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, use_hard_neg=True):
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



