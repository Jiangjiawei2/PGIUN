import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from thop import profile

# def make_model(args, parent=False):
#     model = ISTANetPlus(args)
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f'{total_params:,} total parameters.')
#     total_trainable_params = sum(
#         p.numel() for p in model.parameters() if p.requires_grad)
#     print(f'{total_trainable_params:,} training parameters.')
#     return model
def make_model(args, parent=False):
    a = torch.Tensor(1, 1, 256, 256)
    k = torch.Tensor(1, 1, 256, 256)
    mask = torch.Tensor(1, 1, 256, 256)

    model = ISTANetPlus(args)
    flops, params = profile(model, (a,k,mask,))
    print('flops: ', flops, 'params: ', params)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    return model

class DC_layer_I(nn.Module):
    def __init__(self):
        super(DC_layer_I, self).__init__()

    def forward(self,  x_rec, x_under,mask):
        k_temp= torch.fft.fftshift(torch.fft.fft2(x_rec))
        matrixones = torch.ones_like(mask.data)  # 全为1：(16, 1, 256, 256)
        k_rec_dc = (matrixones - mask) * k_temp + x_under
        out = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(k_rec_dc, dim=(-2, -1))))
        return out  # (16, 1, 256, 256)

# Define ISTA-net-plus block
class BasicBlock(nn.Module):
    def __init__(self, norm='ortho'):
        super(BasicBlock, self).__init__()
        self.norm = norm
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        num_filter = 64

        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(num_filter, 1, 3, 3)))
        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(num_filter, num_filter, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(num_filter, num_filter, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(num_filter, num_filter, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(num_filter, num_filter, 3, 3)))

        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, num_filter, 3, 3)))
        self.dc = DC_layer_I()
    def _forward_operation(self, img, mask):
        k = torch.fft.fftshift(torch.fft.fft2(img, norm=self.norm))
        k = k * mask
        return k

    def _back_operation(self, k, mask):
        k = mask * k
        # img = torch.abs(torch.fft.ifft2(k, norm=self.norm))
        img = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1))))
        return img

    def update_operation(self, f_1, k, mask):
        h_1 = k - self._forward_operation(f_1, mask)
        update = f_1 + self.lambda_step * self._back_operation(h_1, mask)
        return update

    def forward(self, x, k, m):
        x = self.update_operation(x, k, m)

        x_input = self.dc(x, k, m)
        x_D = F.conv2d(x_input, self.conv_D, padding=1)

        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_G = F.conv2d(x_backward, self.conv_G, padding=1)

        x_pred = x_input + x_G

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D
        return x_pred, symloss

class ISTANetPlus(nn.Module):
    def __init__(self, args, n_iter=8, n_convs=5, n_filters=64, norm='ortho'):
        '''
        ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing
        :param n_iter: num of iterations
        :param n_convs:
        :param n_filters:
        :param norm: 'ortho' norm for fft
        '''
        super(ISTANetPlus, self).__init__()

        rec_blocks = []
        self.norm = norm
        self.n_iter = n_iter
        for _ in range(n_iter):
            rec_blocks.append(BasicBlock(self.norm))
        self.rec_blocks = nn.ModuleList(rec_blocks)

    def forward(self, x, k, m):
        layers_sym = []  # for computing symmetric loss
        for i in range(self.n_iter):
            x, layer_sym = self.rec_blocks[i](x, k, m)
            layers_sym.append(layer_sym)
        return x


# if __name__ == '__main__':
#     a = torch.Tensor(1, 1, 48, 48)
#     k = torch.Tensor(1, 1, 48, 48)
#     mask = torch.Tensor(1, 1, 48, 48)
#     out, _ = ISTANetPlus()(a, k, mask)
#     print(a.shape)


