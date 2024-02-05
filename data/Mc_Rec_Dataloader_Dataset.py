# -- coding: utf-8 --
import argparse

import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from os.path import splitext
from tqdm import tqdm
import torch.nn as nn

class Loader:
    def __init__(self, args):
        if args.data_name == 'IXI':
            self.train_dataset = IXI_dataset(args, 'train')
            self.validation_dataset = IXI_dataset(args, 'valid')
            self.test_dataset = IXI_dataset(args, 'test')
            self.args = args
        elif args.data_name == 'fastMRI':
            self.train_dataset = fastMRI_dataset(args, 'train')
            self.validation_dataset = fastMRI_dataset(args, 'valid')
            self.test_dataset = fastMRI_dataset(args, 'test')
            self.args = args
        elif args.data_name == 'BraTS':
            self.train_dataset = BraTS_dataset(args, 'train')
            self.validation_dataset = BraTS_dataset(args, 'valid')
            self.test_dataset = BraTS_dataset(args, 'test')
            self.args = args

    def trainloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.args.batch_size,
                          shuffle=True,
                          pin_memory=not self.args.cpu,
                          num_workers=self.args.n_threads)

    def validloader(self):
        return DataLoader(self.validation_dataset,
                          batch_size=1,
                          shuffle=False,
                          pin_memory=not self.args.cpu,
                          num_workers=self.args.n_threads)

    def testloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          shuffle=False,
                          pin_memory=not self.args.cpu,
                          num_workers=self.args.n_threads)

class IXI_dataset(Dataset):
    """
    PD辅助T2实现重建
    """
    def __init__(self, args, type='train'):
        super(IXI_dataset, self).__init__()
        self.args = args
        self.data_dir = args.data_dir
        self.modal_rec = "T2"
        self.modal_ref = "PD"
        self.acceleration = args.acceleration
        self.type = type

        # 取出数据的主名
        raw_data_path_rec = os.path.join(args.data_dir, args.data_name, 'IXI-Raw', 'IXI-{}'.format(self.modal_rec), type)
        # filenames_rec = sorted(os.listdir(raw_data_path_rec), key=lambda x: x.split('-')[0])
        # filenames_ref = sorted(os.listdir(raw_data_path_ref), key=lambda x: x.split('-')[0])
        filenames_rec = os.listdir(raw_data_path_rec)
        self.filenames_rec = [filename_rec.split('.')[0] for filename_rec in filenames_rec]

        # 加载数据和源码的路径
        ## /home/shx/My_Projects/datasets/IXI/T2/train
        self.space_data_dir_rec = os.path.join(args.data_dir, args.data_name, self.modal_rec, type)
        self.space_data_dir_ref = os.path.join(args.data_dir, args.data_name, self.modal_ref, type)
        ## /home/shx/My_Projects/datasets/IXI/mask/random_mask/X8.py
        self.mask_path = os.path.join(args.data_dir, args.data_name, 'mask', '{}_mask_256'.format(args.mask),
                                      'X{}.npy'.format(self.acceleration))

        self.names_rec = []
        self.names_ref = []
        for i in range(len(self.filenames_rec)):
            for slice in args.slice_indexes:
                self.names_rec.append('{}-{:03d}.npy'.format(self.filenames_rec[i], slice))
                filename_refs = self.filenames_rec[i].split('-')
                filename_ref = '{}-{}-{}-PD'.format(filename_refs[0], filename_refs[1], filename_refs[2])
                self.names_ref.append('{}-{:03d}.npy'.format(filename_ref, slice))

    def __len__(self):
        return len(self.names_rec)

    def __getitem__(self, idx):
        name_rec = self.names_rec[idx]
        # name_refs = splitext(name_rec).split('-')
        name_ref = self.names_ref[idx]

        image_rec = np.load(os.path.join(self.space_data_dir_rec, name_rec))
        image_ref = np.load(os.path.join(self.space_data_dir_ref, name_ref))
        mask = np.load(self.mask_path)

        image_rec = torch.from_numpy(image_rec)
        image_ref = torch.from_numpy(image_ref)
        mask =  torch.from_numpy(mask)

        image_rec = self.norm(image_rec)
        image_ref = self.norm(image_ref)

        fully = torch.fft.fftshift(torch.fft.fft2(image_rec))
        fully_ref = torch.fft.fftshift(torch.fft.fft2(image_ref))
        
        under_sampling = fully * mask

        image_rec = image_rec.unsqueeze(0)
        image_ref = image_ref.unsqueeze(0)
        fully = fully.unsqueeze(0)
        fully_ref = fully_ref.unsqueeze(0)
        under_sampling = under_sampling.unsqueeze(0)
        mask = mask.unsqueeze(0)

        # Separate complex data to two channels data(real and imaginary)
        fully_real = fully.real
        fully_imag = fully.imag
        # fully = torch.cat((fully_real, fully_imag), axis=0)
        fully = torch.cat([fully_real, fully_imag], dim=0)
        fully_ref_real = fully_ref.real
        fully_ref_imag = fully_ref.imag
        # fully_ref = np.concatenate((fully_ref_real, fully_ref_imag), axis=0)
        fully_ref = torch.cat((fully_ref_real, fully_ref_imag), dim=0)

        under_sample_real = under_sampling.real
        under_sample_imag = under_sampling.imag
        # under_sample = np.concatenate((under_sample_real, under_sample_imag), axis=0)
        under_sample = torch.cat((under_sample_real, under_sample_imag), dim=0)

        # return (name_rec,
        #         torch.from_numpy(image_rec),
        #         torch.from_numpy(fully),
        #         torch.from_numpy(under_sample),
        #         torch.from_numpy(mask),
        #         name_ref,
        #         torch.from_numpy(image_ref),
        #         torch.from_numpy(fully_ref))

        return (name_rec,
                image_rec,
                fully,
                under_sample,
                mask,
                name_ref,
                image_ref,
                fully_ref)

    # def norm(self, image_2D):
    #     max_ = np.max(image_2D)
    #     min_ = np.min(image_2D)
    #     if max_ == 0:
    #         return image_2D
    #     return (image_2D - min_) / (max_ - min_)
    def norm(self, image_2D):
        max_ = torch.max(image_2D)
        min_ = torch.min(image_2D)
        if max_ == 0:
            return image_2D
        return (image_2D - min_) / (max_ - min_)


class fastMRI_dataset(Dataset):
    """
    PD（亮一点）辅助FSPD（暗一点）实现重建
    """
    def __init__(self, args, type='train'):
        super(fastMRI_dataset, self).__init__()
        self.args = args
        self.data_dir = args.data_dir
        self.modal_rec = 'FSPD'
        self.modal_ref = 'PD'
        # self.modal_rec = 'PD'
        # self.modal_ref = 'FSPD'

        self.acceleration = args.acceleration
        self.type = type

        raw_data_path_rec = os.path.join(args.data_dir, args.data_name, 'FSPD', type)
        filenames_rec = os.listdir(raw_data_path_rec)
        self.filenames_rec = [filename_rec.split('.')[0] for filename_rec in filenames_rec]
        self.space_data_dir_rec = os.path.join(args.data_dir, args.data_name, self.modal_rec, type)
        self.space_data_dir_ref = os.path.join(args.data_dir, args.data_name, self.modal_ref, type)
        self.mask_path = os.path.join(args.data_dir, args.data_name, 'mask', '{}_mask_320'.format(args.mask),
                                      'X{}.npy'.format(self.acceleration))

        self.filenames = []
        for i in self.filenames_rec:
            num = i.split('-')[1]
            if num not in ['000', '001', '002', '003', '004', '005', '006', '007', '008']:
                self.filenames.append("{}.npy".format(i))
        # print(self.filenames)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name_rec = self.filenames[idx]
        name_ref = self.filenames[idx]
        image_rec = np.load(os.path.join(self.space_data_dir_rec, self.filenames[idx]))
        image_ref = np.load(os.path.join(self.space_data_dir_ref, self.filenames[idx]))

        image_rec = self.norm(image_rec)
        image_ref = self.norm(image_ref)

        fully = np.fft.fftshift(np.fft.fft2(image_rec))
        fully_ref = np.fft.fftshift(np.fft.fft2(image_ref))
        mask = np.load(self.mask_path)
        under_sampling = fully * mask

        image_rec = np.expand_dims(image_rec, axis=0)
        image_ref = np.expand_dims(image_ref, axis=0)
        fully = np.expand_dims(fully, axis=0)
        fully_ref = np.expand_dims(fully_ref, axis=0)
        under_sampling = np.expand_dims(under_sampling, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Separate complex data to two channels data(real and imaginary)
        fully_real = fully.real
        fully_imag = fully.imag
        fully = np.concatenate((fully_real, fully_imag), axis=0)

        fully_ref_real = fully_ref.real
        fully_ref_imag = fully_ref.imag
        fully_ref = np.concatenate((fully_ref_real, fully_ref_imag), axis=0)

        under_sample_real = under_sampling.real
        under_sample_imag = under_sampling.imag
        under_sample = np.concatenate((under_sample_real, under_sample_imag), axis=0)

        return (name_rec,
                torch.from_numpy(image_rec),
                torch.from_numpy(fully),
                torch.from_numpy(under_sample),
                torch.from_numpy(mask),
                name_ref,
                torch.from_numpy(image_ref),
                torch.from_numpy(fully_ref))

    def norm(self, image_2D):
        max_ = np.max(image_2D)
        min_ = np.min(image_2D)
        if max_ == 0:
            return image_2D
        return (image_2D - min_) / (max_ - min_)

class BraTS_dataset(Dataset):
    """
    T1辅助T2重建
    """
    def __init__(self, args, type='train'):
        super(BraTS_dataset, self).__init__()
        self.args = args
        self.data_dir = args.data_dir
        self.modal_rec = 'T2'
        self.modal_ref = 'T1'
        # self.modal_rec = 'PD'
        # self.modal_ref = 'FSPD'

        self.acceleration = args.acceleration
        self.type = type

        raw_data_path_rec = os.path.join(args.data_dir, args.data_name, 'T2', type)
        filenames_rec = os.listdir(raw_data_path_rec)
        self.filenames_rec = [filename_rec.split('.')[0] for filename_rec in filenames_rec]
        self.space_data_dir_rec = os.path.join(args.data_dir, args.data_name, self.modal_rec, type)
        self.space_data_dir_ref = os.path.join(args.data_dir, args.data_name, self.modal_ref, type)
        self.mask_path = os.path.join(args.data_dir, args.data_name, 'mask', '{}_mask_240'.format(args.mask),
                                      'X{}.npy'.format(self.acceleration))

        self.filenames = []
        for i in self.filenames_rec:
            num = i.split('-')[1]
            if num in ['030', '040', '045', '050', '055', '060', '065', '070', '075', '080', '085', '090', '095', '100',
                       '105', '110', '115', '120', '125']:
                self.filenames.append("{}.npy".format(i))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name_rec = self.filenames[idx]
        name_ref = self.filenames[idx]
        image_rec = np.load(os.path.join(self.space_data_dir_rec, self.filenames[idx]))
        image_ref = np.load(os.path.join(self.space_data_dir_ref, self.filenames[idx]))

        image_rec = self.norm(image_rec)
        image_ref = self.norm(image_ref)

        fully = np.fft.fftshift(np.fft.fft2(image_rec))
        fully_ref = np.fft.fftshift(np.fft.fft2(image_ref))
        mask = np.load(self.mask_path)
        under_sampling = fully * mask

        image_rec = np.expand_dims(image_rec, axis=0)
        image_ref = np.expand_dims(image_ref, axis=0)
        fully = np.expand_dims(fully, axis=0)
        fully_ref = np.expand_dims(fully_ref, axis=0)
        under_sampling = np.expand_dims(under_sampling, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Separate complex data to two channels data(real and imaginary)
        fully_real = fully.real
        fully_imag = fully.imag
        fully = np.concatenate((fully_real, fully_imag), axis=0)

        fully_ref_real = fully_ref.real
        fully_ref_imag = fully_ref.imag
        fully_ref = np.concatenate((fully_ref_real, fully_ref_imag), axis=0)

        under_sample_real = under_sampling.real
        under_sample_imag = under_sampling.imag
        under_sample = np.concatenate((under_sample_real, under_sample_imag), axis=0)

        return (name_rec,
                torch.from_numpy(image_rec),
                torch.from_numpy(fully),
                torch.from_numpy(under_sample),
                torch.from_numpy(mask),
                name_ref,
                torch.from_numpy(image_ref),
                torch.from_numpy(fully_ref))

    def norm(self, image_2D):
        max_ = np.max(image_2D)
        min_ = np.min(image_2D)
        if max_ == 0:
            return image_2D
        return (image_2D - min_) / (max_ - min_)
    


if __name__ == '__main__':
    import torch.nn.functional as F
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='test')
    parser.add_argument(
        '--data_dir', type=str, default="F:\\Mc_datasets", help='data_dir'
    )
    parser.add_argument(
        '--acceleration', type=int, default="4", help='accel'
    )
    parser.add_argument(
        '--data_name', type=str, default='fastMRI', help='IXI'
    )
    parser.add_argument(
        '--mask', type=str, default='random', help='random'
    )
    parser.add_argument(
        '--slice_indexes', default=[40, 50, 60, 70, 80]
    )
    args = parser.parse_args()

    train_dataset = fastMRI_dataset(args, 'train')
    trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True, num_workers=6)


    class Edge(nn.Module):
        def __init__(self):
            super(Edge, self).__init__()
            k = torch.Tensor([[.05, .25, .4, .25, .05]])
            self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(1, 1, 1, 1)
            if torch.cuda.is_available():
                self.kernel = self.kernel.cuda()

        def conv_gauss(self, img):
            n_channels, _, kw, kh = self.kernel.shape
            img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
            return F.conv2d(img, self.kernel, groups=n_channels)

        def laplacian_kernel(self, current):
            filtered = self.conv_gauss(current)  # filter
            down = filtered[:, :, ::2, ::2]  # downsample
            new_filter = torch.zeros_like(filtered)
            new_filter[:, :, ::2, ::2] = down * 4  # upsample
            filtered = self.conv_gauss(new_filter)  # filter
            diff = current - filtered
            return diff

        def forward(self, x):

            return self.laplacian_kernel(x)

    edge = Edge()

    for idx, (name_rec, image_rec, fully, under_sample, mask, name_ref, image_ref) in tqdm(enumerate(trainloader), desc='Batch',
                                                              total=len(trainloader), ncols=80):
        # image = edge(image_rec.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

        # plt.imshow(np.array(image.cpu()[0][0]), 'gray')
        # plt.show()
        print(image_rec.size())
        # print("{}--{}".format(name_rec, name_ref))
        # print(torch.min(image_rec))
        # print(fully.shape)
        # print(under_sample.shape)
        # print(mask.shape)
        # print(image_ref.shape)
        # print()









