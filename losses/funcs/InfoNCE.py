import torch
import torch.nn as nn
from collections import OrderedDict

from .loss_func_utils import dot_sim_matrix, cosine_sim_matrix


# class MultiViewCodingLoss(nn.Module):
class InfoNCE(nn.Module):
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
        aux_losses = OrderedDict()
        density_ratios = self.den_ratio_func(view1, view2)#, debug=kwargs['debug'])
        density_ratios = density_ratios/self.tao

        ## View 1 as anchor
        log_sm_den_ratios = -torch.nn.functional.log_softmax(density_ratios, dim=-1) 
        correct = log_sm_den_ratios.diag()
        loss = torch.sum(correct)

        # batch scale
        loss = loss/correct.size(0)
        aux_losses["1->2"] = loss


        
        # ## View 2 as anchor
        log_sm_den_ratios = -torch.nn.functional.log_softmax(density_ratios.transpose(0,1), dim=-1) 
        correct = log_sm_den_ratios.diag()#torch.gather(log_sm_den_ratios, dim=0, index=indexer)
        loss2 = torch.sum(correct)

        # batch scale
        loss2 = loss2/(correct.size(0))

        aux_losses["2->1"] = loss2
        loss = loss + loss2
        return loss, aux_losses



