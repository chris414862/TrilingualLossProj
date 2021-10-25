import torch
import torch.nn as nn

from .loss_func_utils import check_tensor
from steps.utils.general_utils import EPSILON


def hsphere_uniformity_loss(view:torch.Tensor, t=2.0, view_id=""):
    # view dims: [batch, feat_dims]
    sq_distances = torch.pdist(view, p=2).pow(2)
    # sq_distances dims: [batch*(batch-1)/2]
    avg = sq_distances.mul(-t).exp().mean()
    avg = avg + EPSILON
    ret = avg.log()
    if check_tensor(ret, view_id, "ul"):
        check_tensor(view, view_id, "ul input")
        check_tensor(avg, view_id, "ul after avg")
    return ret, None

def hsphere_align_loss(view1, view2, alpha=2.0, view_pair_id=""):
    # view dims: [batch, feat_dims]
    ret = (view1 - view2).norm(p=2, dim=1).pow(alpha).mean()
    if check_tensor(ret, view_pair_id, "al"):
        check_tensor(view1, view_pair_id, "al input 1")
        check_tensor(view2, view_pair_id, "al input 2")
    return ret, None




# class HypersphericLoss():
#     def __init__(self, alpha=2, t=2, align_weight=1., uniform_weight=1.):
#         self.alpha=alpha
#         self.t=t
#         self.align_weight=align_weight
#         self.uniform_weight=uniform_weight
#
#     def align_loss(self, view1, view2):
#         # view dims: [batch, feat_dims]
#         return (view1 - view2).norm(p=2, dim=1).pow(self.alpha).mean()
#
#     def u_single_view(self, view:torch.Tensor):
#         # view dims: [batch, feat_dims]
#         sq_distances = torch.pdist(view, p=2).pow(2)
#         # sq_distances dims: [batch*(batch-1)/2]
#         ret =  sq_distances.mul(-self.t).exp().mean()
#         ret = ret + EPSILON
#         return ret.log()
#
#     def uniformity_loss(self, view1, view2, view_pair_id="", **kwargs):
#         u1 =  self.u_single_view(view1) 
#         u2 =  self.u_single_view(view2)/2
#
#         return u1 + u2, u1, u2
#
#
#     def __call__(self, view1, view2, view_pair_id="",  **kwargs):
#         al =self.align_loss(view1, view2) 
#         al = self.align_weight*al
#         ul, ul1, ul2 = self.uniformity_loss(view1, view2, view_pair_id=view_pair_id)
#         ul = self.uniform_weight*ul
#         if check_tensor(ul, view_pair_id, "ul") or check_tensor(al, view_pair_id, "al"):
#             check_tensor(view1, view_pair_id, "input from first in pair")
#             check_tensor(view2, view_pair_id, "input from second in pair")
#             check_tensor(ul2, view_pair_id, "ul from first")
#             check_tensor(ul2, view_pair_id, "ul from second")
#
#         return al + ul, {"al":al, "ul":ul}




