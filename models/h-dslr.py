
import torch.nn as nn
import torch
from einops import rearrange
import math
import warnings
from torch import einsum
import torch.nn.functional as F
from thop import profile
from timm.models.layers import DropPath, trunc_normal_, drop_path
import numpy as np

def make_model(args, parent=False):
    a = torch.complex(torch.ones(1,1,256, 256), torch.ones(1,1,256, 256))
    # k = torch.Tensor(1, 1, 256, 256)
    mask = torch.Tensor(1, 1, 256, 256)
    # gt = torch.Tensor(1, 1, 256, 256)
    model = SLR()
    flops, params = profile(model, (a,mask,))
    print('flops: ', flops/1e9, 'params: ', params/1e6)

    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(
    #     p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    return model



class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_relu=True):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU() if use_relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class FiveLayerBlock(nn.Module):
    def __init__(self, in_channels, features):
        super(FiveLayerBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, features)
        self.conv2 = ConvLayer(features, features)
        self.conv3 = ConvLayer(features, features)
        self.conv4 = ConvLayer(features, features)
        self.conv5 = ConvLayer(features, in_channels, use_relu=False)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x + residual
        return x

        out = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(x_rec_dc, dim=(-2, -1))))

def jwxjwy(res):
    a,b=get_kspace_inds(np.asarray(res))
    a=a.astype(np.float32)
    b=b.astype(np.float32)  
    a=2*a/256
    b=2*b/256  
    s=np.asarray([2,res[0],res[1]])
    dz=np.zeros(s,dtype=np.complex64)
    dz[0,:,:]=np.reshape(1j*a,res)
    dz[1,:,:]=np.reshape(1j*b,res)
    return dz

def get_kspace_inds(res):
    if res[0] % 2:
        indx=list(np.arange(0,((res[1]-1)/2)+1))
        indx=indx+list(np.arange(-(res[1]-1)/2,0))
    else:
        indx=list(np.arange((-res[1]/2),(res[1]/2)))
        indx = np.fft.ifftshift(indx)
        
    if res[1] % 2:
        indy=list(np.arange(0,((res[0]-1)/2)+1))
        indy=indy+list(np.arange(-(res[0]-1)/2,0))
 
    else:
        indy=list(np.arange((-res[0]/2),(res[0]/2)))
        indy = np.fft.ifftshift(indy)
 

    kx,ky=np.meshgrid(indx,indy)
    kx=np.fft.fftshift(kx)
    ky=np.fft.fftshift(ky)
    kx=np.reshape(kx,(np.size(kx),1))
    ky=np.reshape(ky,(np.size(ky),1))
    return kx,ky

def r2c(x):
# Convert multi-channel data from real to complex in PyTorch
    I_real = x[:, 0:1, :, :]
    I_imag = x[:, 1:2, :, :]
    out = torch.complex(I_real, I_imag)  # bs ,1, 256, 256
    return out

def c2r(k):
    # Convert multi-channel data from complex to real in PyTorch
    x_K_real = k.real
    x_K_imag = k.imag
    out = torch.cat([x_K_real, x_K_imag], dim=1)
    return out

def grad(x, gr):
    # Compute gradient along an axis in PyTorch
    x = r2c(x) * gr
    return c2r(x)

def gradh(x, gr):
    # Compute Hermitian of gradient operation in PyTorch
    x = r2c(x) * torch.conj(gr)
    return c2r(x)

def dc(rhs, mask, lam1, lam2, Glhs):
    lam1 = torch.tensor([lam1], dtype=torch.float32).to(mask.device)
    lam2 = torch.tensor([lam2], dtype=torch.float32).to(mask.device)
    lam1 = torch.complex(lam1, torch.tensor([0.0], dtype=torch.float32).to(mask.device))
    lam2 = torch.complex(lam2, torch.tensor([0.0], dtype=torch.float32).to(mask.device))
    lhs = mask + (lam1 * Glhs) + lam2
    rhs = r2c(rhs)
    output = torch.div(rhs, lhs)
    output = c2r(output)
    return output

class SLR(nn.Module):  ##  out=np.zeros((nImg,1,256,170,2),dtype=dtype)
    def __init__(self, features=64, K=10):
        super(SLR, self).__init__()
        self.fivelkx = FiveLayerBlock(2, features)
        self.fivelky = FiveLayerBlock(2, features)
        self.fivelim = FiveLayerBlock(2, features)
        self.K = K
        self.res= [256,256]

    def forward(self, atb, mask):  
        '''
        atb = undersample // (B, 1, w, h )
        '''
        a=jwxjwy(self.res)
        jwx = a[0]
        jwy = a[1]
        del a
        jwx=np.tile(jwx,[atb.shape[0],1,1])
        jwx=np.expand_dims(jwx,1)
        jwy=np.tile(jwy,[atb.shape[0],1,1])
        jwy=np.expand_dims(jwy,1)

        jwx = torch.tensor(jwx).to(atb.device)
        jwy = torch.tensor(jwy).to(atb.device)

        I_real = atb.real
        I_imag = atb.imag
        k_in = torch.cat([I_real, I_imag], dim=1)

        jwx = torch.tensor(jwx, dtype=torch.complex64)
        jwy = torch.tensor(jwy, dtype=torch.complex64)

        Glhs = (jwx * torch.conj(jwx)) + (jwy * torch.conj(jwy))
        out = {}
        out['dc0'] = k_in

        for i in range(1, self.K + 1):
            j = str(i)

            out['dwkx' + j] = gradh(self.fivelkx(grad(out['dc' + str(i - 1)], jwx)), jwx)
            out['dwky' + j] = gradh(self.fivelky(grad(out['dc' + str(i - 1)], jwy)), jwy)

            temp = out['dc' + str(i - 1)]
            temp =  r2c(temp)
            temp = torch.fft.ifft2(torch.fft.ifftshift(temp, dim=(-2, -1)))
            # temp = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(temp, dim=(-2, -1))))
            x_i =  c2r(temp)

            x_temp = self.fivelim(x_i)
            x_temp =  r2c(x_temp)
            x_temp = torch.fft.fftshift(torch.fft.fft2(x_temp))
            # x_temp = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x_temp, dim=(-2, -1))))

            out['dwim' + j]  = c2r(x_temp)

            lam1 = 1.0
            lam2 = 1.0
            rhs = k_in + out['dwkx' + j] + out['dwky' + j] + out['dwim' + j]
            out['dc' + j] = dc(rhs, mask, lam1, lam2, Glhs)

        outf=r2c(out['dc'+str(self.K)])
        outf = torch.fft.ifft2(torch.fft.ifftshift(outf, dim=(-2, -1)))
        outf = torch.abs(outf)

        return outf



