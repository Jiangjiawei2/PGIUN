'''
        -Quadratic Splitting Network for MR Image Reconstruction
'''
import torch.nn as nn
from collections import OrderedDict
import torch
from thop import profile

def make_model(args, parent=False):
    a = torch.Tensor(1, 1, 256, 256)
    k = torch.Tensor(1, 1, 256, 256)
    mask = torch.Tensor(1, 1, 256, 256)

    model = HQSNet(args)
    flops, params = profile(model, (a,k,mask,))
    print('flops: ', flops/1e9, 'params: ', params/1e6)

    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(
    #     p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    return model

def conv_block(model_name='hqs-net', channel_in=22, n_convs=3, n_filters=32):
    '''
    reconstruction blocks in DC-CNN;
    primal(image)-net and dual(k)-space-net blocks in LPD-net
    regular cnn reconstruction blocks in HQS-Net
    :param model_name: 'dc-cnn', 'prim-net', or 'hqs-net'
    :param channel_in:
    :param n_convs:
    :param n_filters:
    :return:
    '''
    layers = []
    if model_name == 'dc-cnn':
        channel_out = channel_in
    elif model_name == 'prim-net' or model_name =='hqs-net':
        channel_out = channel_in - 1
    elif model_name == 'dual-net':
        channel_out = channel_in - 4

    for i in range(n_convs - 1):
        if i == 0:
            layers.append(nn.Conv2d(channel_in, n_filters, 3, 1, 1))
        else:
            layers.append(nn.Conv2d(n_filters, n_filters, 3, 1, 1))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers.append(nn.Conv2d(n_filters, channel_out, 3, 1, 1))

    return nn.Sequential(*layers)

def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

'''
# --------------------------------------------
# Useful blocks
# https://github.com/xinntao/BasicSR
# --------------------------------
# conv + normaliation + relu (conv)
# resblock (ResBlock)
# --------------------------------------------
'''

# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------

def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR',
         negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(
               nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            )
        elif t =='T':
            L.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-4, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R' or t == 'r':
            L.append(nn.ReLU(inplace=True))
        elif t == 'L' or t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)

# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True,
                 mode='CRC', negative_slope=0.2):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels = out_channels'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        return x + self.res(x)


"""
# --------------------------------------------
# Upsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# upsample_pixelshuffle
# upsample_upconv
# upsample_convtranspose
# --------------------------------------------
"""

# --------------------------------------------
# conv + subp (+ relu)
# --------------------------------------------
def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True,
                          mode='2R', negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, ..., 4BR.'
    up1 = conv(in_channels, out_channels * (int(mode[0])**2), kernel_size, stride, padding, bias, mode='C' + mode,
               negative_slope=negative_slope)
    return up1

# --------------------------------------------
# nearest_upsample + conv (+ R)
# --------------------------------------------
def upsample_upconv(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True,
                  mode='2R', negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, ..., 4BR.'
    if mode[0] == '2':
        uc = 'UC'
    elif mode[0] == '3':
        uc = 'uC'
    elif mode[0] == '4':
        uc = 'vC'
    mode = mode.replace(mode[0], uc)
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode,
               negative_slope=negative_slope)
    return up1

# --------------------------------------------
# convTranspose (+ relu)
# --------------------------------------------
def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0,
                          bias=True, mode='2R', negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return up1

'''
# --------------------------------------------
# Downsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# downsample_strideconv
# downsample_maxpool
# downsample_avgpool
# --------------------------------------------
'''

# --------------------------------------------
# strideconv (+ relu)
# --------------------------------------------
def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True,
                          mode='2R', negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return down1

# --------------------------------------------
# maxpooling + conv (+ relu)
# --------------------------------------------
def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
                       bias=True, mode='2R', negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'MC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0],
                negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode[1:],
                     negative_slope=negative_slope)
    return sequential(pool, pool_tail)

# --------------------------------------------
# averagepooling + conv (+ relu)
# --------------------------------------------
def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='2R',
                       negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'AC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:],
                     negative_slope=negative_slope)
    return sequential(pool, pool_tail)

# used as modified unet reconstruction blocks in HQS-Net
class UNetRes(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                 downsample_mode='strideconv', upsample_mode='contranspose'):
        super(UNetRes, self).__init__()

        self.m_head = conv(in_nc, nc[0], bias=False, mode='C')

        #downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = sequential(
            *[ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=False, mode='2')
        )
        self.m_down2 = sequential(
            *[ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=False, mode='2')
        )
        self.m_down3 =sequential(
            *[ResBlock(nc[2], nc[2], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[2], nc[3], bias=False, mode='2')
        )
        self.m_bocy = nn.Sequential(
            *[ResBlock(nc[3], nc[3], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)]
        )

        if upsample_mode == 'upconv':
            upsample_block = upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError('upsample_mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = sequential(
            upsample_block(nc[3], nc[2], bias=False, mode='2'),
            *[ResBlock(nc[2], nc[2], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)]
        )
        self.m_up2 = sequential(
            upsample_block(nc[2], nc[1], bias=False, mode='2'),
            *[ResBlock(nc[1], nc[1], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)]
        )
        self.m_up1 = sequential(
            upsample_block(nc[1], nc[0], bias=False, mode='2'),
            *[ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode + 'C') for _ in range(nb)]
        )

        self.m_tail = conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        return x

class DC_layer_I(nn.Module):
    def __init__(self):
        super(DC_layer_I, self).__init__()

    def forward(self,  x_rec, x_under,mask):
        k_temp= torch.fft.fftshift(torch.fft.fft2(x_rec))
        matrixones = torch.ones_like(mask.data)  # 全为1：(16, 1, 256, 256)
        k_rec_dc = (matrixones - mask) * k_temp + x_under
        out = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(k_rec_dc, dim=(-2, -1))))
        return out  # (16, 1, 256, 256)

class HQSNet(nn.Module):
    def __init__(self, args, buffer_size=5, n_iter=8, n_convs=6, n_filter=64, block_type='cnn', norm='ortho'):
        '''
        :param buffer_size: m
        :param n_iter: n
        :param n_convs: convolutions in each reconstruction block
        :param n_filter: output channel for convolutions
        :param block_type: 'cnn' or 'unet
        :param norm: 'ortho' norm for fft
        '''
        super(HQSNet, self).__init__()
        self.norm = norm
        self.m = buffer_size
        self.n_iter = n_iter
        # the initialization of mu may influence the final accuracy
        self.mu = nn.Parameter(0.5 * torch.ones((1, 1)))
        self.block_type = block_type
        if self.block_type == 'cnn':
            rec_blocks = []
            for i in range(self.n_iter):
                rec_blocks.append(
                    conv_block('hqs-net', channel_in=self.m + 1, n_convs=n_convs, n_filters=n_filter)
                )
            self.rec_blocks = nn.ModuleList(rec_blocks)
        elif self.block_type == 'unet':
            self.rec_blocks = UNetRes(in_nc= self.m + 1, out_nc=self.m, nc=[64, 128, 256, 512],
                                      nb=4, act_mode='R', downsample_mode='strideconv',
                                      upsample_mode='convtranspose')
        # self.dc = DC_layer_I()
    def _forward(self, img, mask):
        k = torch.fft.fftshift(torch.fft.fft2(img, norm=self.norm))
        k = k * mask
        return k

    def _backward_operation(self, k, mask):
        k = mask * k
        img = torch.abs(torch.fft.ifft2(k, norm=self.norm))
        return img

    def update_operation(self, f_1, k, mask):
        h_1 = k - self._forward(f_1, mask)
        update = f_1 + self.mu * self._backward_operation(h_1, mask)
        return update

    def forward(self, img, k, mask):
        '''
        :param img: zero_filled imgs (batch, 1, h, w)
        :param k: corresponding undersampled k-space data (batch, 1, h, w)
        :param mask: sampling mask
        :return: reconstructed img
        '''
        # initialize buffer f: the concatenation of m copies of the zero-filled images
        f = torch.cat([img] * self.m, 1).to(img.device)

        # n reconstruction blocks
        for i in range(self.n_iter):
            f_1 = f[:, 0:1].clone()
            update_f_1 = self.update_operation(f_1, k, mask)
            # update_f_1 = dc(update_f_1, k, mask)
            if self.block_type == 'cnn':
                f = f + self.rec_blocks[i](torch.cat([f, update_f_1], dim=1))
            else:
                f = f + self.rec_blocks(torch.cat([f, update_f_1], dim=1))
        return f[:, 0:1]

if __name__ == '__main__':
    a = torch.Tensor(1, 1, 48, 48)
    k = torch.Tensor(1, 1, 48, 48)
    mask = torch.Tensor(1, 1, 48, 48)
    # out = HQSNet()(a, k, mask)
    net = HQSNet(args)

    input = torch.randn(1, 3, 112, 112)
    flops, params = profile(net, (a,k,mask,))
    print('flops: ', flops, 'params: ', params)
    print(a.shape)

