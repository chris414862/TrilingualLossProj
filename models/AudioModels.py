# Author: David Harwath, Wei-Ning Hsu
import math
import numpy as np
import torch
import torch.nn as nn
from .CommonLayers import MyMHAttention

from .quantizers import VectorQuantizerEMA, TemporalJitter

class SpeechBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, width=9, stride=1, downsample=None):
        super(SpeechBasicBlock, self).__init__()
        self.conv1 = conv1d(inplanes, planes, width=width, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1d(planes, planes, width=width)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

def make_batch_mask(nframes, max_seq_len, device):
    """
        makes a mask for a single batch.

        Parameters:
        nframes (torch.Tensor [batch])
            - Tensor containing the lengths of each sequence in batch
    """
    nframes = nframes[:, np.newaxis]
    indeces = torch.arange(max_seq_len, device=device)
    bs = nframes.shape[0]
    batch_indeces = indeces.expand(bs, max_seq_len).type(torch.float)
    # batch_indeces dims: [batch_size, max_seq_len]
    bool_mask = torch.lt(batch_indeces , nframes)
    # bool_mask dims: [batch_size, max_seq_len]
    float_mask = bool_mask.type(torch.float)
    return float_mask






class ResDavenet(nn.Module):
    def __init__(self, feat_dim=40, block=SpeechBasicBlock, layers=[2, 2, 2, 2],
                 layer_widths=[128, 128, 256, 512, 1024], convsize=9, output_head="avg", device=None, scale_pe=True):
        assert(len(layers) == 4)
        assert(len(layer_widths) == 5)
        super(ResDavenet, self).__init__()
        self.output_head_str = output_head
        self.feat_dim = feat_dim
        self.inplanes = layer_widths[0]
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=(self.feat_dim,1), 
                               stride=1, padding=(0,0), bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, layer_widths[1], layers[0], 
                                       width=convsize, stride=2)
        self.layer2 = self._make_layer(block, layer_widths[2], layers[1], 
                                       width=convsize, stride=2)
        self.layer3 = self._make_layer(block, layer_widths[3], layers[2], 
                                       width=convsize, stride=2)
        self.layer4 = self._make_layer(block, layer_widths[4], layers[3], 
                                       width=convsize, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if self.output_head_str == "avg":
            # self.pool_func = nn.AdaptiveAvgPool2d((1, 1))
            self.head_layer = self.avg_output
        elif self.output_head_str == "mh_attn":
            self.head_layer = MyMHAttention(layer_widths[-1], nhead=8, seq_len=500, scale_pe=scale_pe)

    def _make_layer(self, block, planes, blocks, width=9, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )       
        layers = []
        layers.append(block(self.inplanes, planes, width=width, stride=stride, 
                            downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, width=width, stride=1))
        return nn.Sequential(*layers)


    def avg_output(self, audio_outputs, nframes=None, **kwargs):
        """
            Avergages embeddings from all time steps.

            Parameters:
            audio_outputs (torch.Tensor [batch, time_steps, embed_size])
                -- output from penultimate layer
        """


        # audio_outputs dims: [batch, time_steps, embed_size]
        if nframes is not None:
            batch_mask = make_batch_mask(nframes, max_seq_len=audio_outputs.size(-2), device=audio_outputs.device)
            batch_mask = batch_mask.unsqueeze(-1)
            audio_outputs = audio_outputs*batch_mask
            nframes = nframes.unsqueeze(-1)
            # nframes dims: [batch, 1]
            divisor = nframes
        else:
            divisor =audio_outputs.size(1) 

        audio_outputs = audio_outputs.sum(dim=1)/divisor
        return audio_outputs

    def forward(self, x, nframes=None, view_id=""):
        def check_tensor(tens, ident, label): 
            if torch.isnan(tens).sum() > 0 or torch.isinf(tens).sum() > 0:
                print(f"AUDIO MODEL: id: {ident} label: {label} nans: {torch.isnan(tens).sum()}, infs: {torch.isinf(tens).sum()}")
                return True
            else:
                return False

        # x dims: [batch, audio_feat_dim, max_time_steps]
        orig_frames = x.size(-1)
        # print("orig x:", x.shape)
        orig_x = x 
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        after_first_conv =x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x dims: [batch, embed_dim, 1, downsampled_time_steps]
        x = x.squeeze(2).transpose(1,2)
        pre_head_x = x
        # x dims: [batch, downsampled_time_steps, embed_dim]
        pooling_ratio = round(orig_frames / x.size(-2))
        if nframes is not None:
            # Behavior of div has changed. Before TF ~1.4 this performed truncated integer division
            curr_nframes:torch.Tensor = torch.div(nframes, pooling_ratio)

            # Check if float was returned (by newer TF version)
            if curr_nframes.dtype != torch.int64:
                curr_nframes = torch.trunc(curr_nframes)

            curr_nframes = curr_nframes.type(torch.float) 
            curr_nframes = torch.where(curr_nframes<=0.,
                                       torch.Tensor([1.]).to(curr_nframes.device).type(curr_nframes.dtype),
                                       curr_nframes) # prevent div by 0
        else:
            curr_nframes = nframes
        x = self.head_layer(x, nframes=curr_nframes)
        
        if check_tensor(x, view_id, "output CHECK"):
            check_tensor(orig_x, view_id, "input")
            check_tensor(after_first_conv, view_id, "output")
            check_tensor(pre_head_x, view_id, "pre-head")
        # x dims: [batch, embed_dim]
        return x

def conv1d(in_planes, out_planes, width=9, stride=1, bias=False):
    """1xd convolution with padding"""
    if width % 2 == 0:
        pad_amt = int(width / 2)
    else:
        pad_amt = int((width - 1) / 2)
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,width), 
                     stride=stride, padding=(0,pad_amt), bias=bias)

def flatten_tensor(inputs):
    """
    Convert a 4D tensor of shape (B, C, H, W) to a 2D tensor of shape 
    (B*H*W, C) and return (B, H, W, C) shape
    """
    inputs = inputs.permute(0, 2, 3, 1).contiguous()
    bhwc = inputs.shape
    return inputs.view(-1, bhwc[-1]), bhwc

def unflatten_tensor(inputs, bhwc):
    """
    Inverse function for flatten_tensor()
    """
    if inputs is None:
        return inputs
    return inputs.view(bhwc).permute(0, 3, 1, 2)

def get_flattened_indices(nframes, padded_len):
    indices = []
    for i, nframe in enumerate(nframes):
        indices.append(torch.arange(nframe) + i * padded_len)
    return torch.cat(indices).to(nframes.device)


            
        


        # a = self.align_mat(x)
        # # a dims: [batch, time_steps, heads]
        # print("MySelfAttn - a:", a.size())
        # a = self.sm(a)
        # # a dims: [batch, time_steps, heads]
        # re_alignments = torch.bmm(torch.transpose(a, 1,2), x)
        # print("MySelfAttn - re_alignments1:", re_alignments.size())
        # # re_alignments: [batch, heads, embed_dim]
        # re_alignments = re_alignments.flatten(-2)
        # print("MySelfAttn - re_alignments2:", re_alignments.size())
        # # re_alignments: [batch, heads*embed_dim]
        # out = self.mix_mat(re_alignments)
        # print("MySelfAttn - out:", out.size())
        # # out: [batch, embed_dim]
        # out = out.unsqueeze(-2)
        # print("MySelfAttn - out:", out.size(),flush=True)
        # # out: [batch, 1, embed_dim]

        # return out

# class MyAudioAvgLayer(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#
#     def forward(self, x, nframes):

# class MyAudioSelfAttn(MySelfAttn):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         
#
#     def forward(self, x, nframes):
#         # x dims: [batch, embed_dim, 1, time_step]
#         x: torch.Tensor = x.squeeze(2)
#         # x dims: [batch, embed_dim, time_step]
#         x = x.transpose(1,2)
#         # x dims: [batch, time_step, embed_dim]
#         x = super().forward(x)
#         # x dims: [batch, embed_dim]
#         return x
#         
# class MyAudioResidual(nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.self_attn = MyAudioSelfAttn(**kwargs)
#         self.avg_output = MyAudioAvgLayer()
#         print("MY AUDIO RESIDUAL")
#
#     def forward(self, x, nframes):
#         out = self.self_attn(x, nframes) + self.avg_output(x, nframes)
#         print("REs out:", out.size())
#         return out


class ResDavenetVQ(ResDavenet):
    def __init__(self, feat_dim=40, block=SpeechBasicBlock, 
                 layers=[2, 2, 2, 2], layer_widths=[128, 128, 256, 512, 1024],
                 convsize=9, codebook_Ks=[512, 512, 512, 512, 512], 
                 commitment_cost=1, jitter_p=0.0, vqs_enabled=[0, 0, 0, 0, 0], 
                 EMA_decay=0.99, init_ema_mass=1, init_std=1, 
                 nonneg_init=False, output_head="avg"):
        assert(len(codebook_Ks) == 5)
        assert(len(vqs_enabled) == 5)

        super().__init__(feat_dim=feat_dim, block=block, layers=layers, 
                         layer_widths=layer_widths, convsize=convsize)
        for l in range(5):
            if vqs_enabled[l]:
                quant_layer = VectorQuantizerEMA(
                        codebook_Ks[l], layer_widths[l], commitment_cost, 
                        decay=EMA_decay, init_ema_mass=init_ema_mass,
                        init_std=init_std, nonneg_init=nonneg_init)
                setattr(self, 'quant%d' % (l + 1), quant_layer)
        self.jitter_p = jitter_p
        self.jitter = TemporalJitter(p_left=jitter_p, p_right=jitter_p)
        self.vqs_enabled = list(vqs_enabled)
        #TODO: Calculate embed_size from layer ops rather than hard code
        # self.self_attn = MySelfAttn(num_heads=8, embed_size=1024, data_len=213, model_str="Audio") 
        self.ln_final = nn.LayerNorm(1024)
        self.bn_final = nn.BatchNorm1d(1024)
        self.output_head_str = output_head
        # elif self.output_head_str == "custom_self_attn":
        #     self.residual_output = False
        #     self.num_heads = 8
        #     if self.residual_output:
        #         self.head_layer = MyAudioResidual(num_heads=self.num_heads, embed_size=1024, data_len=117, model_str="Audio") 
        #     else:
        #         self.head_layer = MyAudioSelfAttn(num_heads=self.num_heads, embed_size=1024, data_len=117, model_str="Audio")


    def forward(self, x, nframes=None):
        """
        If nframes is provided, remove padded parts from quant_losses,
        flat_inputs and flat_onehots. This is useful for training, when EMA 
        only requires pre-quantized inputs and assigned indices. Note that
        jitter() is only applied after VQ-{2,3}.

        Args:
            x (torch.Tensor): Spectral feature batch of shape (B, C, F, T) or 
                (B, F, T).
            nframes (torch.Tensor): Number of frames for each utterance. Shape
                is (B,)
        """
        quant_losses = [None] * 5 # quantization losses by layer
        flat_inputs  = [None] * 5 # flattened pre-quantized inputs by layer
        flat_onehots = [None] * 5 # flattened one-hot codes by layer
        orig_frames = x.size(-1)

        if x.dim() == 3:
            x = x.unsqueeze(1)
        L = x.size(-1)
        device = "cuda" if x.is_cuda else "cpu"

        mask = make_batch_mask(nframes, x.size(-1), device)
        x = x * mask
        cur_nframes = None

        x = self.relu(self.bn1(self.conv1(x)))
        if nframes is not None:
            cur_nframes:torch.Tensor = nframes / round(L / x.size(-1))

        (quant_losses[0], x, flat_inputs[0],
         flat_onehots[0]) = self.maybe_quantize(x, 0, cur_nframes)
        x = self.maybe_jitter(x)

        x = self.layer1(x)
        if nframes is not None:
            cur_nframes = nframes / round(L / x.size(-1))
        (quant_losses[1], x, flat_inputs[1],
         flat_onehots[1]) = self.maybe_quantize(x, 1, cur_nframes)
        x = self.maybe_jitter(x)

        x = self.layer2(x)
        if nframes is not None:
            cur_nframes = nframes / round(L / x.size(-1))
        (quant_losses[2], x, flat_inputs[2],
         flat_onehots[2]) = self.maybe_quantize(x, 2, cur_nframes)

        x = self.layer3(x)
        if nframes is not None:
            cur_nframes = nframes / round(L / x.size(-1))
        (quant_losses[3], x, flat_inputs[3],
         flat_onehots[3]) = self.maybe_quantize(x, 3, cur_nframes)

        x = self.layer4(x)
        if nframes is not None:
            cur_nframes = nframes / round(L / x.size(-1))
        (quant_losses[4], x, flat_inputs[4],
         flat_onehots[4]) = self.maybe_quantize(x, 4, cur_nframes)


        pooling_ratio = round(orig_frames / x.size(-1))
        if x.dim() == 4:
            x = x.squeeze(2)

        print("x dims:", x.size())
        sys.exit()
        x = self.head_layer(x, nframes=nframes, pooling_ratio=pooling_ratio)

        return x, quant_losses, flat_inputs, flat_onehots

    # def avg_output(self,x, nframes):
    #     x_size = x.size()
    #
    #     # x dims: [batch, embed_dim, 1, time_step]
    #     x: torch.Tensor = x.squeeze(2)
    #     # x dims: [batch, embed_dim, time_step]
    #
    #     time_steps = x.size(2)
    #     x = x.sum(dim=2)
    #     # squeezed_x dims: [batch, embed_dim]
    #
    #     if nframes is not None:
    #         nframes = nframes[:,np.newaxis]
    #         x = x/nframes
    #     else:
    #         x = x/time_steps
    #
    #     return x

    def maybe_quantize(self, inputs, quant_idx, nframes=None):
        """
        Wrapper for quantization. Return flat_inputs and 
        flat_onehots for separate EMA codebook updates.

        Args:
            inputs (torch.Tensor): Pre-quantized inputs of shape (B, C, H, W).
            quant_idx (int): Index of the quantization layer to use.
            nframes (torch.Tensor): Lengths of shape (B,) w.r.t. inputs to the
                quantization layer (not the raw nframes to the model).
        Returns:
            flat_losses (torch.Tensor): Quantization loss for each frame. A
                tensor of shape (sum(nframes),)
            quant_inputs (torch.Tensor): Quantized input of shape (B, C, H, W).
            flat_inputs (torch.Tensor): Non-padding input frames. A tensor
                of shape (sum(nframes), C)
            flat_onehots (torch.Tensor): One-hot codes for non-padding input
                frames. A tensor of shape (sum(nframes), K)
        """
        flat_inputs, bhwc = flatten_tensor(inputs)
        ret_flat_inputs = flat_inputs
        if nframes is not None:
            indices = get_flattened_indices(nframes, bhwc[2])
            indices = indices.type(torch.long)
            indices = indices.to(inputs.device)
            ret_flat_inputs = torch.index_select(flat_inputs, 0, indices)

        if not self.vqs_enabled[quant_idx]:
            return None, inputs, ret_flat_inputs, None

        quant_layer = getattr(self, 'quant%d' % (quant_idx + 1))
        flat_losses, quant_inputs, flat_onehots = quant_layer(flat_inputs)
        quant_inputs = unflatten_tensor(quant_inputs, bhwc)
        if nframes is not None:
            flat_losses = torch.index_select(flat_losses, 0, indices)
            flat_onehots = torch.index_select(flat_onehots, 0, indices)


        return flat_losses, quant_inputs, ret_flat_inputs, flat_onehots

    def maybe_jitter(self, inputs):
        return self.jitter(inputs) if self.jitter_p > 0 else inputs




    def _attn_wrapper(self, residual=True):
        if residual == True:
            return lambda x, nframes: self._self_attention(x, nframes) + self._avg_embeddings(x, nframes)
        else:
            return self._self_attention

    def ema_update(self, inputs_by_layer, onehots_by_layer):
        """
        Exponential moving average update for enabled codebooks.

        Args:
            inputs_by_layer (list): A list of five torch.Tensor/None objects, 
                Each tensor is a pre-quantized input batch for a VQ layer.
                Shape is (N, D), where N is the number of frames, D is the 
                dimensionality of code embeddings.
            onehots_by_layer (list): A list of five torch.Tensor/None objects,
                which are onehot codes of shape (N, K), where K is the number
                of codes.
        """
        for quant_idx, is_enabled in enumerate(self.vqs_enabled):
            if not is_enabled:
                continue
            x, c = inputs_by_layer[quant_idx], onehots_by_layer[quant_idx]
            assert(x is not None)
            assert(c is not None)
            quant_layer = getattr(self, 'quant%d' % (quant_idx + 1))
            quant_layer.ema_update(x, c)

    def get_vq_outputs(self, x, layer, unflatten=False):
        """
        Get feature around the specified VQ layer. Jittering is not applied.
        """
        assert(layer in ['quant%d' % (d + 1) for d in range(5)])

        if x.dim() == 3:
            x = x.unsqueeze(1)
        L = x.size(-1)

        def _prepare_return(x, losses, preq_x, onehots):
            if unflatten:
                B, _, H, W = x.size()
                losses = unflatten_tensor(losses, (B, H, W, -1))
                preq_x = unflatten_tensor(preq_x, (B, H, W, -1))
                onehots = unflatten_tensor(onehots, (B, H, W, -1))
            return losses, x, preq_x, onehots

        x = self.relu(self.bn1(self.conv1(x)))
        losses, x, preq_x, onehots = self.maybe_quantize(x, 0)
        if layer == 'quant1':
            return _prepare_return(x, losses, preq_x, onehots)

        for quant_idx in range(1, 5):
            x = getattr(self, 'layer%d' % quant_idx)(x)
            losses, x, preq_x, onehots = self.maybe_quantize(x, quant_idx)
            if layer == 'quant%d' % (quant_idx + 1):
                return _prepare_return(x, losses, preq_x, onehots)

    def get_embedding(self, layer):
        """
        Get VQ embedding at the specified layer.
        """
        assert(hasattr(self, layer))
        return getattr(self, layer).get_embedding()
