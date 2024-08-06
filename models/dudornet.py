
import torch.nn as nn
import torch
from einops import rearrange
import math
import warnings
from torch import einsum
import torch.nn.functional as F
from thop import profile
from timm.models.layers import DropPath, trunc_normal_, drop_path
from models.dudornet.utils2 import AverageMeter, get_scheduler, psnr, get_nonlinearity, DataConsistencyInKspace_I, DataConsistencyInKspace_K, fft2, complex_abs_eval

def make_model(args, parent=False):
    a = torch.Tensor(1, 1, 256, 256)
    k = torch.Tensor(1, 1, 256, 256)
    mask = torch.Tensor(1, 1, 256, 256)

    model = RecurrentModel()
    flops, params = profile(model, (a,k,mask,))
    print('flops: ', flops/1e9, 'params: ', params/1e6)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    return model

## ABLATION 
class DRDN(nn.Module):
    def __init__(self, n_channels, G0, kSize, D=3, C=4, G=32, dilateSet=[1,2,4,4]):
        super(DRDN, self).__init__()

        self.D = D   # number of RDB
        self.C = C   # number of Conv in RDB
        self.kSize = kSize  # kernel size
        self.dilateSet = dilateSet   # dilation setting in SERDRB

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(n_channels, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.SEDRDBs = nn.ModuleList()
        for i in range(self.D):
            self.SEDRDBs.append(
                SEDRDB(growRate0=G0, growRate=G, nConvLayers=C, dilateSet=dilateSet)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        # Up-sampling net
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(G0, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.Conv2d(G, n_channels, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

    def forward(self, x):
        f1 = self.SFENet1(x)
        x = self.SFENet2(f1)

        SEDRDBs_out = []
        for j in range(self.D):
            x = self.SEDRDBs[j](x)
            SEDRDBs_out.append(x)

        x = self.GFF(torch.cat(SEDRDBs_out, 1))
        x += f1

        output = self.UPNet(x)

        return output


# Squeeze&Excite Dilated Residual dense block (SEDRDB) architecture
class SEDRDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, dilateSet, kSize=3):
        super(SEDRDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(SEDRDB_Conv(G0 + c * G, G, dilateSet[c], kSize))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

        # Squeeze and Excitation Layer
        self.SE = SELayer(channel=G0, reduction=16)

    def forward(self, x):
        x1 = self.LFF(self.convs(x))
        x2 = self.SE(x1)
        x3 = x2 + x
        return x3


class SEDRDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, dilate, kSize=3):
        super(SEDRDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=dilate * (kSize - 1) // 2, dilation=dilate, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DC_layer_I(nn.Module):
    def __init__(self):
        super(DC_layer_I, self).__init__()

    def forward(self,  x_rec, x_under,mask):
        k_temp= torch.fft.fftshift(torch.fft.fft2(x_rec))
        matrixones = torch.ones_like(mask.data)  # 全为1：(16, 1, 256, 256)
        k_rec_dc = (matrixones - mask) * k_temp + x_under
        out = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(k_rec_dc, dim=(-2, -1))))
        return out, k_rec_dc  # (16, 1, 256, 256)

class DC_layer_K(nn.Module):
    def __init__(self):
        super(DC_layer_K, self).__init__()

    def forward(self, x_rec, x_under,mask):
        matrixones = torch.ones_like(mask.data)  # 全为1：(16, 1, 256, 256)
        k_rec_dc = (matrixones - mask) * x_rec + x_under
        out = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(k_rec_dc, dim=(-2, -1))))
        return out, k_rec_dc  # (16, 1, 256, 256)


class RecurrentModel(nn.Module):
    def __init__(self):
        super(RecurrentModel, self).__init__()

        self.n_recurrent = 5
        self.net_G_I= DRDN(n_channels=1, G0=32, kSize=3, D=3, C=4, G=32, dilateSet=[1,2,3,3])
        self.net_G_K= DRDN(n_channels=2, G0=32, kSize=3, D=3, C=4, G=32, dilateSet=[1,2,3,3])

        self.dcs_I = DC_layer_I()
        self.dcs_K = DC_layer_K()


    def forward(self, img, tag_kspace_full, mask):
        '''
        :param img: zero_filled imgs (batch, 1, h, w)
        :param tag_kspace_full: corresponding undersampled k-space data (batch, 1, h, w)
        :param mask: sampling mask
        :return: reconstructed img
        ''' 
        I = img
        # print(I.shape)
        I.requires_grad_(True)
        net = {}
        for i in range(1, self.n_recurrent + 1):
            x_I = I
            net['r%d_img_pred' % i] = self.net_G_I(x_I)
            net['r%d_img_dc_pred' % i], recon_img_dc_k= self.dcs_I(net['r%d_img_pred' % i], tag_kspace_full, mask)

            '''K Space''' 
            x_K_real = recon_img_dc_k.real
            x_K_imag = recon_img_dc_k.imag
            recon_img_dc_k = torch.cat([x_K_real, x_K_imag], dim=1)

            net['r%d_kspc_pred' % i] = self.net_G_K(recon_img_dc_k)  # output recon kspace

            I_real = torch.unsqueeze(net['r%d_kspc_pred' % i][:, 0, :, :], 1)
            I_imag = torch.unsqueeze(net['r%d_kspc_pred' % i][:, 1, :, :], 1)
            I_temp = torch.complex(I_real, I_imag)  # bs ,1, 256, 256

            I,_  = self.dcs_K(I_temp, tag_kspace_full, mask)  # output data consistency images
            # I = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(I_temp, dim=(-2, -1))))
            # torch.abs(torch.fft.ifft2(torch.fft.ifftshift(x_rec_dc, dim=(-2, -1))))

        self.net = net
        self.recon = I
        return self.net, self.recon



        


