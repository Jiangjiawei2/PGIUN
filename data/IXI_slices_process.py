# -- coding: utf-8 --
import SimpleITK as sitk
import os
import numpy as np
import glob
from tqdm import tqdm
from os.path import splitext
import scipy.io as sio
import matplotlib.pyplot as plt
import argparse

"""

"""


def data_extraction_save(args, type):
    print('>>>Start Processing {} {} sets'.format('T2 and PD', type))
    # F:\\Mc_datasets\\IXI\\IXI-Raw\\IXI-T2\\test
    raw_data_dir_T2 = os.path.join(args.data_dir, args.data_name, '{}-Raw'.format(args.data_name),
                                '{}-{}'.format(args.data_name, 'T2'), type)
    # F:\Mc_datasets\\IXI\\T1\\train
    save_dir_T2 = os.path.join(args.data_dir, args.data_name, 'T2', type)

    # F:\\Mc_datasets\\IXI\\IXI-Raw\\IXI-PD\\test
    raw_data_dir_PD = os.path.join(args.data_dir, args.data_name, '{}-Raw'.format(args.data_name),
                                   '{}-{}'.format(args.data_name, 'PD'), type)
    # F:\Mc_datasets\\IXI\\T1\\train
    save_dir_PD = os.path.join(args.data_dir, args.data_name, 'PD', type)

    print("\tT2 is saved in {}".format(save_dir_T2))
    print("\tPD is saved in {}".format(save_dir_PD))

    if not os.path.exists(save_dir_T2):
        os.makedirs(save_dir_T2, exist_ok=True)

    if not os.path.exists(save_dir_PD):
        os.makedirs(save_dir_PD, exist_ok=True)

    names_T2 = os.listdir(raw_data_dir_T2)
    names_PD = os.listdir(raw_data_dir_PD)
    sum_ = 0
    for name_T2 in tqdm(names_T2, desc='number of raw data', ncols=80):  # name: IXI392-Guys-1064-T1.nii.gz
        fname_T2 = splitext(splitext(name_T2)[0])[0]  # IXI392-Guys-1064-T2
        f_path_T2 = os.path.join(raw_data_dir_T2, name_T2)
        itk_img_T2 = sitk.Cast(sitk.ReadImage(f_path_T2), sitk.sitkFloat64)
        img_array_3D_T2 = sitk.GetArrayFromImage(itk_img_T2)

        fname_PDs = fname_T2.split('-')  # IXI392-Guys-1064-T2
        fname_PD = '{}-{}-{}-PD'.format(fname_PDs[0], fname_PDs[1], fname_PDs[2])
        f_path_PD = os.path.join(raw_data_dir_PD, fname_PD + '.nii.gz')
        itk_img_PD = sitk.Cast(sitk.ReadImage(f_path_PD), sitk.sitkFloat64)
        img_array_3D_PD = sitk.GetArrayFromImage(itk_img_PD)

        # 对三维数据进行切片
        slices_T2 = img_array_3D_T2.shape[0]
        slices_PD = img_array_3D_PD.shape[0]

        if slices_PD == slices_T2:
            if slices_PD >= args.data_slices:
            #     sum_ += 1
            #     print(fname_T2)
                for slice in range(slices_PD):
                    image_2D_T2 = img_array_3D_T2[slice, :, :]
                    np.save(os.path.join(save_dir_T2, fname_T2 + '-{:03d}.npy'.format(slice)), image_2D_T2)
                    image_2D_PD = img_array_3D_PD[slice, :, :]
                    np.save(os.path.join(save_dir_PD, fname_PD + '-{:03d}.npy'.format(slice)), image_2D_PD)
            else:
                continue
    print(sum_)
    # print('>>>Finish {} {} sets\n'.format(args.modal, type))


def main(args):
    data_extraction_save(args, 'train')
    data_extraction_save(args, 'valid')
    data_extraction_save(args, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='slices and normalize')
    parser.add_argument(
        "--data_dir", type=str, default="F:\\Mc_datasets",
        help='data directory'
    )
    parser.add_argument(
        '--data_name', type=str, default='IXI',
        help='data name: IXI, fastMRI, hcp'
    )
    parser.add_argument(
        '--data_ext', type=str, default='npys',
        help='data extension name: npys, mats..'
    )
    parser.add_argument(
        '--data_slices', type=int, default=120,
        help='data slice range: T1: 150, T2: 130'
    )
    args = parser.parse_args()
    main(args)
