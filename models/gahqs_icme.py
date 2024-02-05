'''
        -Quadratic Splitting Network for MR Image Reconstruction
'''
import torch.nn as nn
import torch
from einops import rearrange
import math
import warnings
from torch import einsum
import torch.nn.functional as F
from thop import profile

def make_model(args, parent=False):
    a = torch.Tensor(1, 1, 256, 256)
    k = torch.Tensor(1, 1, 256, 256)
    mask = torch.Tensor(1, 1, 256, 256)

    model = GAHQSNet(args)
    flops, params = profile(model, (a,k,mask,))
    print('flops: ', flops/1e9, 'params: ', params/1e6)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    return model

## ABLATION 
class CSE_Block(nn.Module):
    def __init__(self, in_ch, r):
        super(CSE_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, int(in_ch / r), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(int(in_ch / r), in_ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        s = self.layer(x)
        return s * x



def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

# class PGSAModule(nn.Module):

#     def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 8]):
#         super(PGSAModule, self).__init__()
#         self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
#                             stride=stride, groups=conv_groups[0])
#         self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
#                             stride=stride, groups=conv_groups[1])
#         self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
#                             stride=stride, groups=conv_groups[2])
#         self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
#                             stride=stride, groups=conv_groups[3])
#         self.se_avg = SEWeightModule(planes // 4, mode= 'avg')
#         self.se_max = SEWeightModule(planes // 4, mode = 'max')
#         self.split_channel = planes // 4
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         batch_size = x.shape[0]
#         x1 = self.conv_1(x)
#         x2 = self.conv_2(x)
#         x3 = self.conv_3(x)
#         x4 = self.conv_4(x)

#         feats = torch.cat((x1, x2, x3, x4), dim=1)
#         feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

#         x1_se_avg = self.se_avg(x1)
#         x2_se_avg = self.se_avg(x2)
#         x3_se_avg = self.se_avg(x3)
#         x4_se_avg = self.se_avg(x4)

#         x1_se_max= self.se_max(x1)
#         x2_se_max= self.se_max(x2)
#         x3_se_max = self.se_max(x3)
#         x4_se_max = self.se_max(x4)

#         x_se_avg = torch.cat((x1_se_avg, x2_se_avg, x3_se_avg, x4_se_avg), dim=1)
#         x_se_max = torch.cat((x1_se_max, x2_se_max, x3_se_max, x4_se_max), dim=1)
#         x_se = torch.sigmoid( x_se_avg +  x_se_max)
#         attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
#         attention_vectors = self.softmax(attention_vectors)
#         feats_weight = feats * attention_vectors
#         for i in range(4):
#             x_se_weight_fp = feats_weight[:, i, :, :]
#             if i == 0:
#                 out = x_se_weight_fp
#             else:
#                 out = torch.cat((x_se_weight_fp, out), 1)

#         return out
class PGSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 8]):
        super(PGSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes//8, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//8, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//8, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//8, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])

        self.conv_12 = conv(inplans, planes//8, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_22 = conv(inplans, planes//8, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_32 = conv(inplans, planes//8, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_42 = conv(inplans, planes//8, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])

        # self.se_avg = SEWeightModule(planes // 4, mode= 'avg')
        # self.se_max = SEWeightModule(planes // 4, mode = 'max')
        # self.split_channel = planes // 4
        # self.softmax = nn.Softmax(dim=1)

        self.se_avg = SEWeightModule(planes // 8, mode= 'avg')
        self.se_max = SEWeightModule(planes // 8, mode = 'max')
        self.split_channel = planes // 8
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        x12 = self.conv_1(x)
        x22 = self.conv_2(x)
        x32 = self.conv_3(x)
        x42 = self.conv_4(x)

        
        # feats = torch.cat((x1, x2, x3, x4), dim=1)
        # feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        feats = torch.cat((x1, x2, x3, x4,x12,x22,x32,x42), dim=1)
        feats = feats.view(batch_size, 8, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se_avg = self.se_avg(x1)
        x2_se_avg = self.se_avg(x2)
        x3_se_avg = self.se_avg(x3)
        x4_se_avg = self.se_avg(x4)
        x12_se_avg = self.se_avg(x12)
        x22_se_avg = self.se_avg(x22)
        x32_se_avg = self.se_avg(x32)
        x42_se_avg = self.se_avg(x42)


        x1_se_max= self.se_max(x1)
        x2_se_max= self.se_max(x2)
        x3_se_max = self.se_max(x3)
        x4_se_max = self.se_max(x4)
        x12_se_max= self.se_max(x12)
        x22_se_max= self.se_max(x22)
        x32_se_max = self.se_max(x32)
        x42_se_max = self.se_max(x42)

        # x1_se_avg = self.se_avg(x1)
        # x2_se_avg = self.se_avg(x2)
        # x3_se_avg = self.se_avg(x3)
        # x4_se_avg = self.se_avg(x4)

        # x1_se_max= self.se_max(x1)
        # x2_se_max= self.se_max(x2)
        # x3_se_max = self.se_max(x3)
        # x4_se_max = self.se_max(x4)
        x_se_avg = torch.cat((x1_se_avg, x2_se_avg, x3_se_avg, x4_se_avg,x12_se_avg,x22_se_avg,x32_se_avg,x42_se_avg), dim=1)
        x_se_max = torch.cat((x1_se_max, x2_se_max, x3_se_max, x4_se_max,x12_se_max,x22_se_max,x32_se_max,x42_se_max), dim=1)

        # x_se_avg = torch.cat((x1_se_avg, x2_se_avg, x3_se_avg, x4_se_avg), dim=1)
        # x_se_max = torch.cat((x1_se_max, x2_se_max, x3_se_max, x4_se_max), dim=1)
        x_se = torch.sigmoid( x_se_avg +  x_se_max)
        attention_vectors = x_se.view(batch_size, 8, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(8):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out

class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=4, mode='avg'):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.mode = mode 
    def forward(self, x):
        if self.mode == 'avg':
            out = self.avg_pool(x)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out)
            weight = self.sigmoid(out)
        elif self.mode == 'max':
            out = self.max_pool(x)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out)
            weight = self.sigmoid(out)

        return weight

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)        

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class HS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            dim_head=28,
            heads=8,
            only_local_branch=False
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.only_local_branch = only_local_branch

        # position embedding
        if only_local_branch:
            seq_l = window_size[0] * window_size[1]
            self.pos_emb = nn.Parameter(torch.Tensor(1, heads, seq_l, seq_l))
            trunc_normal_(self.pos_emb)
        else:
            seq_l1 = window_size[0] * window_size[1]
            self.pos_emb1 = nn.Parameter(torch.Tensor(1, 1, heads//2, seq_l1, seq_l1))
            h,w = 256//self.heads,256//self.heads
            seq_l2 = h*w//seq_l1
            self.pos_emb2 = nn.Parameter(torch.Tensor(1, 1, heads//2, seq_l2, seq_l2))
            trunc_normal_(self.pos_emb1)
            trunc_normal_(self.pos_emb2)

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x.shape
        w_size = self.window_size
        assert h % w_size[0] == 0 and w % w_size[1] == 0, 'fmap dimensions must be divisible by the window size'
        if self.only_local_branch:
            x_inp = rearrange(x, 'b (h b0) (w b1) c -> (b h w) (b0 b1) c', b0=w_size[0], b1=w_size[1])
            q = self.to_q(x_inp)
            k, v = self.to_kv(x_inp).chunk(2, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
            q *= self.scale
            sim = einsum('b h i d, b h j d -> b h i j', q, k)
            sim = sim + self.pos_emb
            attn = sim.softmax(dim=-1)
            out = einsum('b h i j, b h j d -> b h i d', attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            out = self.to_out(out)
            out = rearrange(out, '(b h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                            b0=w_size[0])
        else:
            q = self.to_q(x)
            k, v = self.to_kv(x).chunk(2, dim=-1)
            q1, q2 = q[:,:,:,:c//2], q[:,:,:,c//2:]
            k1, k2 = k[:,:,:,:c//2], k[:,:,:,c//2:]
            v1, v2 = v[:,:,:,:c//2], v[:,:,:,c//2:]

            # local branch
            q1, k1, v1 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                              b0=w_size[0], b1=w_size[1]), (q1, k1, v1))
            q1, k1, v1 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads//2), (q1, k1, v1))
            q1 *= self.scale
            sim1 = einsum('b n h i d, b n h j d -> b n h i j', q1, k1)
            sim1 = sim1 + self.pos_emb1
            attn1 = sim1.softmax(dim=-1)
            out1 = einsum('b n h i j, b n h j d -> b n h i d', attn1, v1)
            out1 = rearrange(out1, 'b n h mm d -> b n mm (h d)')

            # non-local branch
            q2, k2, v2 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                                 b0=w_size[0], b1=w_size[1]), (q2, k2, v2))
            q2, k2, v2 = map(lambda t: t.permute(0, 2, 1, 3), (q2.clone(), k2.clone(), v2.clone()))
            q2, k2, v2 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads//2), (q2, k2, v2))
            q2 *= self.scale
            sim2 = einsum('b n h i d, b n h j d -> b n h i j', q2, k2)
            sim2 = sim2 + self.pos_emb2
            attn2 = sim2.softmax(dim=-1)
            out2 = einsum('b n h i j, b n h j d -> b n h i d', attn2, v2)
            out2 = rearrange(out2, 'b n h mm d -> b n mm (h d)')
            out2 = out2.permute(0, 2, 1, 3)

            out = torch.cat([out1,out2],dim=-1).contiguous()
            out = self.to_out(out)
            out = rearrange(out, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                            b0=w_size[0])
        return out

class HSAB(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, HS_MSA(dim=dim, window_size=window_size, dim_head=dim_head, heads=heads, only_local_branch=(heads==1))),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class Double_Conv(nn.Module):
    def __init__(self, in_ch, out_ch, padding, droupt=False):
        super(Double_Conv, self).__init__()
        if droupt:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, padding, dropout=False, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1)
            )
        else:
            self.up = nn.Sequential(
                nn.Conv2d(in_ch, 4 * in_ch, kernel_size=3, stride=1, padding=padding),
                nn.PixelShuffle(2),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1)
            )
        self.conv = Double_Conv(in_ch, out_ch, padding, dropout)

    def forward(self, x1, x2):
        """
        :param x1:  上一个输出的结果
        :param x2:  用来cat的
        :return:
        """
        x1 = self.up(x1)
        diffX = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)

        x1 = F.pad(x1, (math.ceil(diffY / 2), int(diffY / 2), math.ceil(diffX / 2), int(diffX / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x        

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, padding, dropout=False):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            Double_Conv(in_ch, out_ch, padding, dropout)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class PositionalEncodingFourier(nn.Module):
    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)

        return pos

class HST(nn.Module):
    def __init__(self, in_dim=28, out_dim=28, dim=28, num_blocks=[1,1,1] ):
        super(HST, self).__init__()
        self.dim = dim
        self.scales = len(num_blocks)
        split = 4
        width = max(int(math.ceil(in_dim / split)), int(math.floor(in_dim // split)))

        self.width = width
        if split == 1:
            self.nums = 1
        else:
            self.nums = split - 1
        convs = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width))
        self.convs = nn.ModuleList(convs)
        self.norm = nn.BatchNorm2d(in_dim)
        self.pos_embd = PositionalEncodingFourier(dim=in_dim)
        
        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_scale = dim
        for i in range(self.scales-1):
            self.encoder_layers.append(nn.ModuleList([
                HSAB(dim=dim_scale, num_blocks=num_blocks[i], dim_head=dim, heads=dim_scale // dim),
                # nn.Conv2d(dim_scale, dim_scale * 2, 4, 2, 1, bias=False),
                Down(dim_scale, dim_scale * 2, 1),
            ]))
            dim_scale *= 2

        # Bottleneck
        self.bottleneck = HSAB(dim=dim_scale, dim_head=dim, heads=dim_scale // dim, num_blocks=num_blocks[-1])
        self.SE = CSE_Block(dim_scale, 8)
        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.scales-1):
            self.decoder_layers.append(nn.ModuleList([
                # nn.ConvTranspose2d(dim_scale, dim_scale // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                # nn.Conv2d(dim_scale, dim_scale // 2, 1, 1, bias=False),
                Up(dim_scale,dim_scale // 2,1),
                HSAB(dim=dim_scale // 2, num_blocks=num_blocks[self.scales - 2 - i], dim_head=dim,
                     heads=(dim_scale // 2) // dim),
                CSE_Block(dim_scale // 2, 8),
            ]))
            dim_scale //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        x = torch.cat((out, spx[self.nums]), 1)

        pos_encoding = self.pos_embd(b, h_inp, w_inp)
        x = x + pos_encoding
        x = self.norm(x)
        # Embedding
        fea = self.embedding(x)
        # x = x[:,:28,:,:]

        # Encoder
        fea_encoder = []
        for (HSAB, FeaDownSample) in self.encoder_layers:
            fea = HSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, HSAB, CSE) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea, fea_encoder[self.scales-2-i])
            fea = HSAB(fea)
            fea = CSE(fea)

        # Mapping
        out = self.mapping(fea) + x 
        # out = out 
        return out[:, :, :h_inp, :w_inp]

class DC_layer(nn.Module):
    def __init__(self):
        super(DC_layer, self).__init__()

    def forward(self, mask, x_rec, x_under):
        matrixones = torch.ones_like(mask.data)  # 全为1：(16, 1, 256, 256)
        x_rec_dc = (matrixones - mask) * x_rec + x_under
        out = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(x_rec_dc, dim=(-2, -1))))
        return out  # (16, 1, 256, 256)

class GAHQSNet(nn.Module):
    def __init__(self, args, buffer_size=32, n_iter=8, n_filter=64, block_type='cnn', norm='ortho'):
        '''
        :param buffer_size: m
        :param n_iter: n
        :param n_filter: output channel for convolutions
        :param norm: 'ortho' norm for fft
        '''
        super(GAHQSNet, self).__init__()
        self.norm = norm
        self.m = buffer_size
        self.n_iter = n_iter
        # the initialization of mu may influence the final accuracy
        self.eta = nn.Parameter(0.5 * torch.ones((buffer_size, 1)))
        self.beta = nn.Parameter(0.5 * torch.ones((buffer_size, 1)))
        self.DC = DC_layer()
        self.denoiser = nn.ModuleList([])
        self.InfoFusion = nn.ModuleList([])
        self.InfoFusion2 = nn.ModuleList([])
        # self.con2 = nn.Conv2d(buffer_size, buffer_size, kernel_size=3, stride=1, padding=1)
        for _ in range(n_iter):
            self.denoiser.append(
                HST(in_dim=buffer_size, out_dim=buffer_size, dim=20, num_blocks=[1,1,1]),
                # nn.Conv2d(buffer_size, buffer_size, kernel_size=3, stride=1, padding=1),
            )

            self.InfoFusion.append(
                # PGSAModule(buffer_size*3, buffer_size, stride=1, conv_kernels=[3, 5, 7, 9], conv_groups=[1, 4, 8, 8]),

                PGSAModule(buffer_size*3, buffer_size, stride=1, conv_kernels=[3, 5, 7, 9], conv_groups=[1, 4, 4, 4]),
                # nn.Conv2d(buffer_size*3, buffer_size, kernel_size=3, stride=1, padding=1),
            )

            self.InfoFusion2.append(
                # PGSAModule(buffer_size*2, buffer_size, stride=1, conv_kernels=[3, 5, 7, 9], conv_groups=[1, 4, 8, 8]),
                PGSAModule(buffer_size*2, buffer_size, stride=1, conv_kernels=[3, 5, 7, 9], conv_groups=[1, 4, 4, 4]),
            )

    def forward(self, img, y, mask):
        '''
        :param img: zero_filled imgs (batch, 1, h, w)
        :param y: corresponding undersampled k-space data (batch, 1, h, w)
        :param mask: sampling mask
        :return: reconstructed img
        ''' 
        
        # initialize buffer f: the concatenation of m copies of the zero-filled images
        x_0 = torch.cat([img] * self.m, 1).to(img.device)
        y_0 = torch.cat([y] * self.m, 1).to(img.device)
        z_k = []
        z_k.append(x_0)
        z_hat = x_0

        FTy = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(y_0, dim=(-2, -1))))
        # n reconstruction blocks
        for i in range(self.n_iter):
            FZ_hat = torch.fft.fftshift(torch.fft.fft2(z_hat),dim=(-2, -1))
            
            x_temp = torch.cat([z_hat, self.eta[i]*FTy, - self.eta[i]*self.DC(mask, FZ_hat,  y)], dim = 1)
            # x_temp= self.con2(x_temp)
            x_k = self.InfoFusion[i](x_temp)
            z_temp =  self.denoiser[i](x_k)
            z_k.append(z_temp)
            z_hat = self.InfoFusion2[i]( torch.cat([(1+self.beta[i])*z_k[-1], -self.beta[i]*z_k[-2] ], dim=1))

        return x_temp[:, 0:1]




## ablation 1  accelerated


# class GAHQSNet(nn.Module):
#     def __init__(self, args, buffer_size=32, n_iter=8, n_filter=64, block_type='cnn', norm='ortho'):
#         '''
#         :param buffer_size: m
#         :param n_iter: n
#         :param n_filter: output channel for convolutions
#         :param norm: 'ortho' norm for fft
#         '''
#         super(GAHQSNet, self).__init__()
#         self.norm = norm
#         self.m = buffer_size
#         self.n_iter = n_iter
#         # the initialization of mu may influence the final accuracy
#         self.eta = nn.Parameter(0.5 * torch.ones((buffer_size, 1)))
#         self.beta = nn.Parameter(0.5 * torch.ones((buffer_size, 1)))
#         self.DC = DC_layer()
#         self.denoiser = nn.ModuleList([])
#         self.InfoFusion = nn.ModuleList([])
#         # self.InfoFusion2 = nn.ModuleList([])
#         for _ in range(n_iter):
#             self.denoiser.append(
#                 HST(in_dim=buffer_size, out_dim=buffer_size, dim=20, num_blocks=[1,1,1]),
#             )

#             self.InfoFusion.append(
#                 PGSAModule(buffer_size*3, buffer_size, stride=1, conv_kernels=[3, 5, 7, 9], conv_groups=[1, 4, 8, 8]),
#             )

#     def forward(self, img, y, mask):
#         '''
#         :param img: zero_filled imgs (batch, 1, h, w)
#         :param y: corresponding undersampled k-space data (batch, 1, h, w)
#         :param mask: sampling mask
#         :return: reconstructed img
#         ''' 
        
#         # initialize buffer f: the concatenation of m copies of the zero-filled images
#         x_0 = torch.cat([img] * self.m, 1).to(img.device)
#         y_0 = torch.cat([y] * self.m, 1).to(img.device)
#         z_k = []
#         z_k.append(x_0)
#         z_hat = x_0

#         FTy = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(y_0, dim=(-2, -1))))
#         # n reconstruction blocks
#         for i in range(self.n_iter):
#             FZ_hat = torch.fft.fftshift(torch.fft.fft2(z_hat),dim=(-2, -1))

#             x_temp = torch.cat([z_hat, self.eta[i]*FTy, -self.eta[i]*self.DC(mask, FZ_hat,  y)], dim = 1)
#             x_k = self.InfoFusion[i](x_temp)
#             z_hat =  self.denoiser[i](x_k)
#             # z_k.append(z_temp)
#             # z_hat = self.InfoFusion2[i]( torch.cat([(1+self.beta[i])*z_k[-1], -self.beta[i]*z_k[-2] ], dim=1))

#         return x_temp[:, 0:1]

## ablation 2
# class GAHQSNet(nn.Module):
#     def __init__(self, args, buffer_size=32, n_iter=8, n_filter=64, block_type='cnn', norm='ortho'):
#         '''
#         :param buffer_size: m
#         :param n_iter: n
#         :param n_filter: output channel for convolutions
#         :param norm: 'ortho' norm for fft
#         '''
#         super(GAHQSNet, self).__init__()
#         self.norm = norm
#         self.m = buffer_size
#         self.n_iter = n_iter
#         # the initialization of mu may influence the final accuracy
#         self.eta = nn.Parameter(0.5 * torch.ones((buffer_size, 1)))
#         self.beta = nn.Parameter(0.5 * torch.ones((buffer_size, 1)))
#         self.DC = DC_layer()
#         self.denoiser = nn.ModuleList([])
#         self.InfoFusion = nn.ModuleList([])
#         # self.InfoFusion2 = nn.ModuleList([])
#         for _ in range(n_iter):
#             self.denoiser.append(
#                 HST(in_dim=buffer_size, out_dim=buffer_size, dim=20, num_blocks=[1,1,1]),
#             )

#             self.InfoFusion.append(nn.ModuleList([
#                 # PGSAModule(buffer_size*3, buffer_size, stride=1, conv_kernels=[3, 5, 7, 9], conv_groups=[1, 4, 8, 8]),
#                 conv(buffer_size*3, buffer_size, 3,1,1),
#                 CSE_Block(buffer_size, 8),
#             ]))

#     def forward(self, img, y, mask):
#         '''
#         :param img: zero_filled imgs (batch, 1, h, w)
#         :param y: corresponding undersampled k-space data (batch, 1, h, w)
#         :param mask: sampling mask
#         :return: reconstructed img
#         ''' 
        
#         # initialize buffer f: the concatenation of m copies of the zero-filled images
#         x_0 = torch.cat([img] * self.m, 1).to(img.device)
#         y_0 = torch.cat([y] * self.m, 1).to(img.device)
#         z_k = []
#         z_k.append(x_0)
#         z_hat = x_0

#         FTy = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(y_0, dim=(-2, -1))))
#         # n reconstruction blocks
#         for i in range(self.n_iter):
#             FZ_hat = torch.fft.fftshift(torch.fft.fft2(z_hat),dim=(-2, -1))

#             x_temp = torch.cat([z_hat, self.eta[i]*FTy, -self.eta[i]*self.DC(mask, FZ_hat,  y)], dim = 1)
#             for (conv,info) in self.InfoFusion:
#                 x_k = conv(x_temp)
#                 x_k = info(x_k)
#             z_hat =  self.denoiser[i](x_k)
#             # z_k.append(z_temp)
#             # z_hat = self.InfoFusion2[i]( torch.cat([(1+self.beta[i])*z_k[-1], -self.beta[i]*z_k[-2] ], dim=1))

#         return x_temp[:, 0:1]



## ablation 3
# class In_Conv(nn.Module):
#     def __init__(self, in_ch, out_ch, padding):
#         super(In_Conv, self).__init__()
#         self.conv = Double_Conv(in_ch, out_ch, padding)

#     def forward(self, x):
#         x = self.conv(x)
#         return x
# class Out_Conv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(Out_Conv, self).__init__()
#         self.conv = Double_Conv(in_ch, out_ch, 1)

#     def forward(self, x):
#         x = self.conv(x)
#         return x       
# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes):
#         super(UNet, self).__init__()
#         self.inc = In_Conv(n_channels, 32, 1)
#         self.down1 = Down(32, 64, 1)
#         self.down2 = Down(64, 128, 1)
#         self.up2 = Up(128, 64, 1)
#         self.up1 = Up(64, 32, 1)
#         self.outc = Out_Conv(32, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x = self.up2(x3, x2)
#         x = self.up1(x, x1)
#         x = self.outc(x)
#         return x

# class GAHQSNet(nn.Module):
#     def __init__(self, args, buffer_size=32, n_iter=8, n_filter=64, block_type='cnn', norm='ortho'):
#         '''
#         :param buffer_size: m
#         :param n_iter: n
#         :param n_filter: output channel for convolutions
#         :param norm: 'ortho' norm for fft
#         '''
#         super(GAHQSNet, self).__init__()
#         self.norm = norm
#         self.m = buffer_size
#         self.n_iter = n_iter
#         # the initialization of mu may influence the final accuracy
#         self.eta = nn.Parameter(0.5 * torch.ones((buffer_size, 1)))
#         self.beta = nn.Parameter(0.5 * torch.ones((buffer_size, 1)))
#         self.DC = DC_layer()
#         self.denoiser = nn.ModuleList([])
#         self.InfoFusion = nn.ModuleList([])
#         self.InfoFusion2 = nn.ModuleList([])
#         for _ in range(n_iter):
#             self.denoiser.append(
#                 UNet(buffer_size,buffer_size),
#             )

#             self.InfoFusion.append(
#                 PGSAModule(buffer_size*3, buffer_size, stride=1, conv_kernels=[3, 5, 7, 9], conv_groups=[1, 4, 8, 8]),
#             )

#             self.InfoFusion2.append(
#                 PGSAModule(buffer_size*2, buffer_size, stride=1, conv_kernels=[3, 5, 7, 9], conv_groups=[1, 4, 8, 8]),
#             )

#     def forward(self, img, y, mask):
#         '''
#         :param img: zero_filled imgs (batch, 1, h, w)
#         :param y: corresponding undersampled k-space data (batch, 1, h, w)
#         :param mask: sampling mask
#         :return: reconstructed img
#         ''' 
        
#         # initialize buffer f: the concatenation of m copies of the zero-filled images
#         x_0 = torch.cat([img] * self.m, 1).to(img.device)
#         y_0 = torch.cat([y] * self.m, 1).to(img.device)
#         z_k = []
#         z_k.append(x_0)
#         z_hat = x_0

#         FTy = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(y_0, dim=(-2, -1))))
#         # n reconstruction blocks
#         for i in range(self.n_iter):
#             FZ_hat = torch.fft.fftshift(torch.fft.fft2(z_hat),dim=(-2, -1))

#             x_temp = torch.cat([z_hat, self.eta[i]*FTy, -self.eta[i]*self.DC(mask, FZ_hat,  y)], dim = 1)
#             x_k = self.InfoFusion[i](x_temp)
#             z_temp =  self.denoiser[i](x_k)
#             z_k.append(z_temp)
#             z_hat = self.InfoFusion2[i]( torch.cat([(1+self.beta[i])*z_k[-1], -self.beta[i]*z_k[-2] ], dim=1))

#         return x_temp[:, 0:1]


# if __name__ == '__main__':
#     a = torch.Tensor(1, 1, 48, 48)
#     k = torch.Tensor(1, 1, 48, 48)
#     mask = torch.Tensor(1, 1, 48, 48)
#     out = HQSNet()(a, k, mask)
#     print(a.shape)

