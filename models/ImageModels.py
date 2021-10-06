# Author: David Harwath, Wei-Ning Hsu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
from .CommonLayers import MyMHAttention

class MyImageAvgLayer(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        # x dims: [batch, embed_dim, height, width]
        dims = x.size(3)*x.size(2)
        x = x.sum(3).sum(2)/dims
        # x dims: [batch, embed_dim]
        x = x
        # x dims: [batch, embed_dim]
        return x


# class MyImageSelfAttn(MySelfAttn):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         
#
#     def forward(self, x):
#         # x dims: [batch, embed_dim, height, width]
#         x = x.flatten(-2)
#         # x dims: [batch, embed_dim, height*width]
#         x = x.transpose(1,2)
#         # x dims: [batch, height*width, embed_dim]
#         x = super().forward(x)
#         # x dims: [batch, embed_dim]
#         return x
#
# class MyImageResidual(nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.self_attn = MyImageSelfAttn(**kwargs)
#         self.avg_output = MyImageAvgLayer()
#
#     def forward(self, x):
#
#         out = self.self_attn(x) + self.avg_output(x)
#         print("REs out:", out.size())
#         return out

class Resnet50(imagemodels.ResNet):
    def __init__(self, embedding_dim=1024, pretrained=False, output_head="avg"):
        super(Resnet50, self).__init__(imagemodels.resnet.Bottleneck, [3, 4, 6, 3])
        if pretrained:
            model_url = imagemodels.resnet.model_urls['resnet50']
            self.load_state_dict(model_zoo.load_url(model_url))
        self.avgpool = None
        self.output_head_str = output_head
        self.fc = None
        self.embedder = nn.Conv2d(2048, embedding_dim, kernel_size=1, stride=1, padding=0)
        self.ln_final = nn.LayerNorm(1024)
        self.bn_final = nn.BatchNorm1d(1024)
        self.output_head = output_head
        # self.pool_func = nn.AdaptiveAvgPool2d((1, 1))
        if self.output_head_str == "avg":
            self.head_layer = self.avg_output
        elif self.output_head_str == "transformer":
            self.head_layer = MyMHAttention(embedding_dim, nhead=8, seq_len=50)
        # elif self.output_head == "custom_self_attn":
        #     self.residual_output = True
        #     self.num_heads = 8
        #     if self.residual_output:
        #         self.head_layer = MyImageResidual(num_heads=self.num_heads, embed_size=1024) 
        #     else:
        #         self.head_layer = MyImageSelfAttn(num_heads=self.num_heads, embed_size=1024) 

    def avg_output(self, x):
        # x dims: [batch,  height*width, embed_dim]
        x = x.sum(dim=1)/x.size(1)
        # x dims: [batch, embed_dim]
        return x

    def forward(self, x, device=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedder(x)
        # x dims: [batch, embed_dim, height, width]
        x = x.flatten(-2).transpose(1,2)
        # x dims: [batch, height*width, embed_dim]
        x = self.head_layer(x)
        # x dims: [batch, embed_dim]
        return x
