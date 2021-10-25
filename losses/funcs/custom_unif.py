import torch
import sys
from collections import defaultdict, OrderedDict
import math


def unif_cross_entropy(all_pairs):
    logp = logp_func(all_pairs, dim=-1)
    uniform_dist = torch.full(logp.shape,1/logp.shape[-1]).to(device=sim_mat.device)
    # uniform_dist dims [bs*(bs-1)/2,]
    loss = -1*(logp* uniform_dist).sum()
    return loss

def gauss_kernel_dist(all_pairs):
    avg = (all_pairs**2).mul(-2).exp().mean()
    avg = avg + EPSILON
    loss = avg.log()
    return loss

def get_all_pairs(sim_mat):
    idx = torch.LongTensor([[j for j in range(sim_mat.shape[0]) if j != i] for i in range(sim_mat.shape[0])]).to(device=sim_mat.device)
    no_eye_sim_mat = torch.gather(sim_mat, 1, idx)
    all_pairs = torch.triu(no_eye_sim_mat).flatten()
    return all_pairs


def custom_unif_loss(view, args):
    """
        Loss to encourage uniformity as measured by some similarity score.
        Unlike the hyperspheric uniformity loss, this function does not assume
        that inputs are normalized.
    """
    bs = view.shape[0]

    # Functions for log(softmax(..))
    if args.custom_unif_sim == "cosine":
        sim_mat_func = cos_sim_matrix
    elif args.custom_unif_sim == "dot":
        sim_mat_func = dot_sim_matrix
    # default
    else:
        sim_mat_func = dot_sim_matrix

    if args.custom_unif_loss == "unif_ce":
        loss_measure_func = torch.nn.functional.log_softmax
    elif args.custom_unif_loss == "neg_entropy":
        loss_measure_func = torch.nn.functional.log_softmax
    elif args.custom_unif_loss == "gauss_kern":
        loss_measure_func = gauss_kernel_dist
    elif args.custom_unif_loss == "hspheric":
        sim_mat_func = cos_sim_matrix
        loss_measure_func = gauss_kernel_dist




    sim_mat = sim_mat_func(view, view)

    # get all pairs w/o pairing with self
    all_pairs = get_all_pairs(sim_mat)
    # all_pairs dims [bs*(bs-1)/2,]

    loss = loss_measure_func(all_pairs)


    return loss
