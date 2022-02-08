import torch
import torch.nn as nn
from collections import OrderedDict

from .loss_func_utils import dot_sim_matrix, cosine_sim_matrix


# class MultiViewCodingLoss(nn.Module):
class InfoNCE(nn.Module):
    def __init__(self, embed_size=None, sim_measure="cosine", batch_size=None, temperature=1.0, masked_margin=False, scheduler=False, adaptive=False):
        super().__init__()
        self.embed_size = embed_size
        self.similarity_measure = sim_measure
        self.epsilon = 1e-8 # from Pytorch implementation (I assume for numerical stability)
        self.tao = temperature #.1  from Mulit-view Coding paper
        self.batch_size = batch_size
        self.masked_margin = masked_margin
        self.adaptive = adaptive
        if scheduler:
            if not adaptive:
                self.scheduler = self.orig_exponential_increase
                self.base = .001
                self.multiplier = 1.002
                self.increase_after = 1000
            else: #use adaptive masked margin (AMM)
                self.alpha = .5
        else:
            self.scheduler = self.ident#lambda x, y: x


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

    def orig_exponential_increase(self, margin, step, **kwargs):
        # margin dims: [bs, bs]
        # margin should be identity mat
        new_margin = self.base*self.multiplier**(step//self.increase_after ) * margin
        return new_margin

    def ident(self, x, **kwargs):
        return x

    def adaptive_margin(self, margin, density_ratios, axis=1,**kwargs):
        # margin dims: [bs, bs]
        # margin should be identity mat
        # eye = torch.eye(margin.shape[0], margin.shape[1]).to(margin.device)
        rev_eye = margin*-1 +1 # flips the zeros and ones in eye
        adaptive_M = (margin*density_ratios)# isolate positive pairs
        adaptive_M = self.alpha * (adaptive_M - (rev_eye*density_ratios).sum(axis=axis)/(margin.shape[0]-1)) # calculation from AMM paper
        return margin* adaptive_M # assure that a diagonal matrix is returned (should be redundant since adaptive_M above is a diag mat)
    

    def bilinear_transform(self, view1, view2):
        """
        This is more similar to the implementation in the infoNCE paper
        """

        #TODO: Finish this function
        # view1, view2 dims: [batch,embed_size]
        den_ratios = self.bilin_mat(view1, view2)
        # den_ratios dims: [batch,embed_size]




    def forward(self, view1, view2, step=None, **kwargs):
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
        aux_losses = OrderedDict()
        density_ratios = self.den_ratio_func(view1, view2)#, debug=kwargs['debug'])
        # density_ratios dims: [bs, bs]
        density_ratios = density_ratios/self.tao
        orig_density_ratios = density_ratios*1
        if self.masked_margin:
            margin = torch.eye(density_ratios.shape[0], density_ratios.shape[1]).to(view1.device)
            # margin dims: [bs, bs]

            if not self.adaptive:
                margin = self.scheduler(margin, step=step)
            else:
                margin = self.adaptive_margin(margin, density_ratios=density_ratios, axis=1)

            density_ratios = density_ratios - margin

        ## View 1 as anchor
        log_sm_den_ratios = -torch.nn.functional.log_softmax(density_ratios, dim=-1) 
        correct = log_sm_den_ratios.diag()
        loss = torch.sum(correct)

        # batch scale
        loss = loss/correct.size(0)
        aux_losses["1->2"] = loss


        
        # ## View 2 as anchor
        if self.masked_margin and self.adaptive:
            margin = torch.eye(orig_density_ratios.shape[0], orig_density_ratios.shape[1]).to(view1.device)
            # Switch the axis that gets summed in the adaptive masked margin. Notice the transpose in the next log_softmax
            margin = self.adaptive_margin(margin, density_ratios=orig_density_ratios, axis=0)
            density_ratios = orig_density_ratios - margin


        log_sm_den_ratios = -torch.nn.functional.log_softmax(density_ratios.transpose(0,1), dim=-1) 
        correct = log_sm_den_ratios.diag()#torch.gather(log_sm_den_ratios, dim=0, index=indexer)
        loss2 = torch.sum(correct)

        # batch scale
        loss2 = loss2/(correct.size(0))

        aux_losses["2->1"] = loss2
        loss = loss + loss2
        return loss, aux_losses



