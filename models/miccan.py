import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_


def make_model(args):
    model = MICCAN(1, 1, args)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\t{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f'\t{total_trainable_params:,} training parameters.')
    return model

# MICCAN with long skip-connection
class MICCAN(nn.Module):
    def __init__(self, in_ch, out_ch, args):
        super(MICCAN, self).__init__()
        # if args.get_patch and args.train == 'train':
        #     input_size = args.patch_size
        # else:
        input_size = 256
        self.layer = nn.ModuleList([UNetCSE(in_ch, out_ch) for _ in range(5)])
        self.dc = DC_layer()
        self.n_blocks = 5
        self.conv_last = nn.Conv2d(out_ch, 1, kernel_size=(3, 3), padding=1, stride=(1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_(m.weight.data, 0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, under_image_rec, under_sample, mask):
        x_rec_dc = under_sample  # (16,1,256,256)
        for i in range(self.n_blocks - 1):
            x_rec = self.layer[i](x_rec_dc)
            x_res = x_rec_dc + x_rec
            x_rec_dc = self.dc(mask, x_res, under_sample)
        x_rec = self.layer[i + 1](x_rec_dc)
        x_res = under_sample + x_rec
        k_rec = self.dc(mask, x_res, under_sample)

        # Output as an image (Space)
        k_rec_isf = torch.fft.ifftshift(k_rec, dim=(-2, -1))
        image_rec_if = torch.fft.ifft2(k_rec_isf)
        image_rec = torch.abs(image_rec_if)
        image_rec = self.conv_last(image_rec)

        return image_rec  # 返回空域的image


class DC_layer(nn.Module):
    def __init__(self):
        super(DC_layer, self).__init__()

    def forward(self, mask, x_rec, x_under):
        matrixones = torch.ones_like(mask.data)  # 全为1：(16, 1, 256, 256)
        x_rec_dc = (matrixones - mask) * x_rec + x_under
        return x_rec_dc  # (16, 1, 256, 256)


# 2 convs
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


# 4 convs
class Cascade_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Cascade_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_ch, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


# 2 convs
class In_Conv(nn.Module):
    def __init__(self, in_ch, out_ch, padding):
        super(In_Conv, self).__init__()
        self.conv = Double_Conv(in_ch, out_ch, padding)

    def forward(self, x):
        x = self.conv(x)
        return x


# 2 convs
class Out_Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Out_Conv, self).__init__()
        self.conv = Double_Conv(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


# mp +2 convs
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


# up + conv
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


# UNet
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = In_Conv(n_channels, 32, 1)
        self.down1 = Down(32, 64, 1)
        self.down2 = Down(64, 128, 1)
        self.up2 = Up(128, 64, 1)
        self.up1 = Up(64, 32, 1)
        self.outc = Out_Conv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up2(x3, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x


# CA
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


# UNet with channel-wise attention
class UNetCSE(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNetCSE, self).__init__()
        self.inc = In_Conv(in_channels, 32, 1)
        self.down1 = Down(32, 64, 1)
        self.down2 = Down(64, 128, 1)
        self.se3 = CSE_Block(128, 8)
        self.up2 = Up(128, 64, 1)
        self.se2 = CSE_Block(64, 8)
        self.up1 = Up(64, 32, 1)
        self.se1 = CSE_Block(32, 8)
        self.outc = Out_Conv(32, n_classes)

    def forward(self, x_rec_dc):
        # x_rec_dc: k-space data
        x_under_ifs = torch.fft.ifftshift(x_rec_dc, dim=(-2, -1))
        x_image = torch.fft.ifft2(x_under_ifs)
        # x_image_real = x_image.real  # (16, 1, 256, 256)
        # x_image_imag = x_image.imag  # (16, 1, 256, 256)
        # x_image_2c = torch.cat([x_image_real, x_image_imag], dim=1)  # (16, 2, 256, 256), 0:real. 1: imaginary
        x_image_2c = torch.abs(x_image)  # (16, 1, 256, 256)
        x1 = self.inc(x_image_2c)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = self.se3(x3)
        x = self.up2(x3, x2)
        x = self.se2(x)
        x = self.up1(x, x1)
        x = self.se1(x)
        x = self.outc(x)  # (16, 1, 256, 256)
        # x_image_real = torch.unsqueeze(x[:, 0, :, :], 1)
        # x_image_imag = torch.unsqueeze(x[:, 1, :, :], 1)
        # x_image = torch.complex(x_image_real, x_image_imag)
        x = torch.fft.fft2(x)
        x = torch.fft.fftshift(x, dim=(-2, -1))
        # 返回的也是k-space data
        return x  # (16, 1, 256, 256)


if __name__ == '__main__':
    from option import args

    model = make_model(args)
    b = torch.Tensor(6, 1, 256, 256)
    b = model(b, b)
    print(b.shape)
