import torch
import torch.nn as nn
import math
import numpy as np





class BYOL_Layer(nn.Module):
    def __init__(self, input_size, layer_output_sizes=[4096, 256]): 
        super().__init__()
        layers = self._make_seq_layers(input_size, layer_output_sizes)
        self.seq_layer = nn.Sequential(*layers)

    def _make_seq_layers(self, input_size, layer_output_sizes=[4096, 256]): 
        layers = []
        prev_size = input_size
        for i, size in enumerate(layer_output_sizes):
            layers.append(nn.Linear(prev_size, size))
            # Last layer has no BatchNorm according to BYOL paper
            if i == len(layer_output_sizes) -1:
                layers.append(nn.BatchNorm2d(size))
            layers.append(nn.Relu)
            prev_size = size
        
        return layers


    def forward(self, x):
        return self.seq_layer(x)



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500, scaled=True):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Unsqueeze the second dimension for the outer product
        position = torch.arange(max_len).unsqueeze(1).type(torch.float)
        # position dims: [max_len, 1]

        div_term = torch.exp(torch.arange(0, self.d_model, 2).type(torch.float).unsqueeze(0)*(-math.log(10000.0) / self.d_model))
        # div_term dims: [1, d_model/2]

        # 1 is for the batch dimension
        self.pe = nn.Parameter(torch.zeros(1, max_len, self.d_model, requires_grad=False), requires_grad=False)
        # self.pe dims: [1, max_len, d_model]

        raw_positions = torch.mm(position, div_term)
        # raw_positions dims: [max_len, d_model/2]

        if scaled:
            scale = 1/math.sqrt(self.d_model)
        else:
            scale = 1
        self.pe[0, :, 0::2] = self.pe[0, :, 0::2] + torch.sin(raw_positions)*scale
        self.pe[0, :, 1::2] = self.pe[0, :, 1::2] + torch.cos(raw_positions)*scale
        # self.pe dimes: [1, max_len, d_model]

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [ batch_size, seq_len, embedding_dim]
        """
        x =x + self.pe[:,:x.size(1)]
        return self.dropout(x)


class MyMHAttention(nn.Module):

    def __init__(self, d, nhead, seq_len, scale_pe):
        super(MyMHAttention, self).__init__()
        self.pos_enc = PositionalEncoding(d_model=d, max_len=seq_len, scaled=scale_pe)
        self.cls_embed = nn.Embedding(1, d)
        # self.mh_layer = torch.nn.MultiheadAttention(d_model=d, nhead=nhead)#, batch_first=True)
        self.mh_layer = torch.nn.MultiheadAttention(d, nhead)#, batch_first=True)
        self.bn_final = nn.BatchNorm1d(1024)
        self.ln_final = nn.LayerNorm(1024)

        # self.mh_layer = torch.nn.MultiheadAttention(nhead=nhead)#, batch_first=True)

    def forward(self, outputs, **kwargs):
        # outputs dims: [batch, seq_len, embed]

        # add dummy 'CLS' token
        cls_idxs = torch.LongTensor(np.zeros((outputs.size(0)))).to(outputs.device) 
        dummy_cls = self.cls_embed(cls_idxs)
        # dummy_cls dims: [batch, embed_size]
        dummy_cls = dummy_cls.unsqueeze(1)
        # dummy_cls dims: [batch, 1, embed_size]
        outputs = torch.cat((dummy_cls,outputs), dim=1)
        # outputs dims: [batch, time_steps+1, embed_dim]
        outputs = self.pos_enc(outputs)
        # outputs dims: [batch, time_steps+1, embed_dim]

        outputs = outputs.transpose(0,1)
        # outputs dims: [ time_steps+1,batch, embed_dim]
        outputs, _ = self.mh_layer(outputs, outputs, outputs)
        # outputs dims: [ time_steps+1,batch, embed_dim]

        outputs = outputs[0,:,:]
        # outputs = self.bn_final(outputs)
        # outputs dims: [batch, embedding]
        return outputs




class My3dLinear(nn.Module):
    def __init__(self, num_heads, num_concepts, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.num_concepts = num_concepts
        self.head_dim = head_dim

        self.concepts_mat = nn.Parameter(torch.zeros((self.num_heads, self.head_dim, self.num_concepts)))
        self.concepts_mat.requires_grad = True
        self.concepts_mat = nn.init.kaiming_normal_(self.concepts_mat)
        self.concepts_bias = nn.Parameter(torch.zeros((1, self.num_concepts)))
        self.concepts_bias.requires_grad = True
        self.concepts_bias = nn.init.kaiming_normal_(self.concepts_bias)


    def forward(self, x):
        # x dims: [batch, num_heads, data_records, head_dim]
        x = torch.matmul(x, self.concepts_mat)
        # x dims: [batch, num_heads, data_records, hidden_concept_classes]
        x = x + self.concepts_bias
        # x dims: [batch, num_heads, data_records, hidden_concept_classes]
        return x

        


class MySelfAttn(nn.Module):
    def __init__(self, num_heads=2, embed_size=1024, data_len=49, hidden_concept_classes=128, ff_hidden_dim=1024*8, silent=False, mask=None, model_str=""):
        super().__init__()
        #TODO: Apply mask, positional encoding
        
        self.num_heads = num_heads 
        self.embed_size = embed_size
        self.head_dim = embed_size//num_heads
        self.ff_hidden_dim = ff_hidden_dim
        self.headed_ff_dim = self.ff_hidden_dim//self.num_heads
        self.hidden_concept_classes = self.head_dim#hidden_concept_classes
        self.silent = silent
        self.model_str = model_str
        
        self.Q_mat = nn.Linear(self.embed_size, self.embed_size)
        self.V_mat = nn.Linear(self.embed_size, self.embed_size)
        self.K_mat = nn.Linear(self.embed_size, self.embed_size)

        # self.concept_mats = My3dLinear(self.num_heads, self.hidden_concept_classes, self.head_dim)
        self.sm = nn.Softmax(dim=-1)
        # self.headed_ff = nn.Linear(self.hidden_concept_classes*self.head_dim, self.headed_ff_dim)
        self.headed_ff = nn.Linear(self.headed_ff_dim, self.ff_hidden_dim)
        print("headed_ff:", self.headed_ff.weight.shape)
        self.gelu = nn.GELU()
        self.ff = nn.Linear(self.ff_hidden_dim, self.embed_size)
        print("ff:", self.ff.weight.shape)

    def project_w_heads(self, x, lin_layer):
        bs = x.size(0)

        print(self.model_str+" - Project_w_heads x:", x.size()) if not self.silent else ""
        # x dims: [batch, time_steps, embed_dim]
        projection = lin_layer(x)
        print(self.model_str+" - Project_w_heads projection:", projection.size()) if not self.silent else ""
        # projection dims: [batch, time_steps, embed_dim]
        headed_proj: torch.Tensor = projection.reshape(bs, -1, self.num_heads, self.head_dim)
        print(self.model_str+" - Project_w_heads headed_proj:", headed_proj.size()) if not self.silent else ""
        # headed_proj dims: [batch, time_steps, num_heads, head_dim]
        out = headed_proj.permute(0,2,1,3)
        print(self.model_str+" - Project_w_heads out:", out.size()) if not self.silent else ""
        # out dims: [batch,  num_heads, time_steps, head_dim]

        return out


    def forward(self, x:torch.Tensor, mask=None):
        print(self.model_str+" - MySelfAttn - x:", x.size()) if not self.silent else ""
        # x dims: [batch, time_steps, embed_dim]
        headed_Q = self.project_w_heads(x, self.Q_mat)
        print(self.model_str+" - MySelfAttn - headed_Q:", headed_Q.size()) if not self.silent else ""
        # headed_Q dims: [batch,  num_heads, time_steps, head_dim]
        headed_V = self.project_w_heads(x, self.V_mat)
        print(self.model_str+" - MySelfAttn - headed_V:", headed_V.size()) if not self.silent else ""
        # headed_V dims: [batch,  num_heads, time_steps, head_dim]
        headed_K = self.project_w_heads(x, self.K_mat)
        # headed_K dims: [batch,  num_heads, time_steps, head_dim]
        print(self.model_str+" - MySelfAttn - headed_K:", headed_K.size(), headed_K.grad_fn) if not self.silent else ""
        # headed_V dims: [batch,  num_heads, time_steps, head_dim]

        # concept_scores = self.concept_mats(headed_Q)
        concept_scores = torch.matmul(headed_Q, headed_K.transpose(2,3))
        print("headed_K.T:", headed_K.transpose(2,3).size(), headed_K.transpose(2,3).grad_fn)
        print(self.model_str+" - MySelfAttn - concept_scores:", concept_scores.size(), concept_scores.grad_fn) if not self.silent else ""
        ### concept_scores dims: [batch, num_heads, time_steps, hidden_concept_classes]
        # concept_scores dims: [batch, num_heads, time_steps, time_steps]

        concept_alignments = self.sm(concept_scores)
        if not mask is None:
            print("HOLD UP!!!")
            concept_alignments = concept_alignments*mask
        # concept_alignments = concept_alignments.transpose(2,3)
        print(self.model_str+" - MySelfAttn - concept_alignments:", concept_alignments.size(), concept_alignments.grad_fn) if not self.silent else ""
        #### concept_alignments dims: [batch, num_heads, hidden_concept_classes, time_steps]
        # concept_alignments dims: [batch, num_heads, time_steps, time_steps(scores)]

        head_concept_vectors = torch.matmul(concept_alignments, headed_V)
        print(self.model_str+" - MySelfAttn - head_concept_vectors:", head_concept_vectors.size(), head_concept_vectors.grad_fn) if not self.silent else ""
        ##### head_concept_vectors dims: [batch, num_heads, hidden_concept_classes, head_dim]
        # head_concept_vectors dims: [batch, num_heads, time_steps, head_dim]

        flattened_concept_vecs = head_concept_vectors.transpose(1,2).flatten(-2)
        print(self.model_str+" - MySelfAttn - flattened_concept_vecs:", flattened_concept_vecs.size(), flattened_concept_vecs.grad_fn) if not self.silent else ""
        #### flattened_concept_vecs dims: [batch, num_heads, hidden_concept_classes*head_dim]
        #TODO: Fix the shape error here. Time steps should be in dim 1
        # flattened_concept_vecs dims: [batch,time_steps, num_heads*head_dim]

        head_vectors = self.headed_ff(flattened_concept_vecs)
        head_vectors = self.gelu(head_vectors)
        print(self.model_str+" - MySelfAttn - head_vectors:", head_vectors.size(), head_vectors.grad_fn) if not self.silent else ""
        # head_vectors dims: [batch, time_steps, hidden_ff_dim] 

        flattened_head_vecs = head_vectors.flatten(-2)
        oooooo
        o
        y

        t
        t
        t

        print(self.model_str+" - MySelfAttn - flattened_head_vecs:", flattened_head_vecs.size()) if not self.silent else ""
        # flattened_head_vecs dims: [batch, time_steps, ] 

        final_representation = self.ff(flattened_head_vecs)
        print(self.model_str+" - MySelfAttn - final_representation:", final_representation.size()) if not self.silent else ""
        # final_representation dims: [batch, embed_dim]

        return final_representation
