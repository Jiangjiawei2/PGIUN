import numpy as np
import torch
import torch.nn as nn
import os

from tqdm import tqdm
import torchvision.utils as vutils
from torch.nn.utils import clip_grad_norm

import data.Mc_Rec_Dataloader_Dataset as Dataloader_Dataset

from utils.averagemeter import AverageMeter
from utils.save_image import save_image

from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import normalized_root_mse
import matplotlib.pyplot as plt
from utils.loss import CharbonnierLoss, PerceptualLoss, EdgeLoss
import csv
n_iter = 0



def train(train_loader, model, loss, optimizer, epoch, train_writer, device, logger, args):
    global n_iter
    train_losses = AverageMeter()
    loss_ = torch.FloatTensor([0.]).to(device)
    loss_.requires_grad_(True)
    loss_sr = torch.FloatTensor([0.]).to(device)
    loss_sr.requires_grad_(True)
    loss_rec = torch.FloatTensor([0.]).to(device)
    loss_rec.requires_grad_(True)

    # ------------------------------------------------------------------------------------------------------------------
    # total image numbers
    if args.data_name == 'IXI':
        total = len(Dataloader_Dataset.IXI_dataset(args, 'train'))
    elif args.data_name == 'fastMRI':
        total = len(Dataloader_Dataset.fastMRI_dataset(args, 'train'))

    model.train()

    # ------------------------------------------------------------------------------------------------------------------
    logger.info('Epoch [{}/{}]\tLearning Rate: {:.3e}'.format(
        epoch, args.n_epochs, optimizer.state_dict()['param_groups'][0]['lr']))
    for idx, (name_rec, image_rec, fully, under_sample, mask, name_ref, image_ref, fully_ref) in enumerate(train_loader):
        image_rec = image_rec.float().to(device)  # (bs, 1, 256, 256)
        fully = fully.float().to(device)  # (bs, 2, 256, 256)
        under_sample = under_sample.float().to(device)  # (bs, 2, 256, 256)
        mask = mask.float().to(device)  # (bs, 1, 256, 256)
        image_ref = image_ref.float().to(device)
        fully_ref = fully_ref.float().to(device)  # (bs, 2, 256, 256)


        # # --------------------------------------------------------------------------------------------------------------

        # 将under_sample变为复数形式
        under_sample_real = torch.unsqueeze(under_sample[:, 0, :, :], 1)
        under_sample_imag = torch.unsqueeze(under_sample[:, 1, :, :], 1)
        under_sample = torch.complex(under_sample_real, under_sample_imag)  # bs ,1, 256, 256
        under_image_rec = torch.abs(torch.fft.ifft2(under_sample))

        image_Rec, x_rec_lift,x_under_lift,x_gt_lift = model(under_image_rec, under_sample, mask,image_rec)

    
        loss_ = lossc(image_Rec, image_rec) + 0.1*(lossc(x_rec_lift,x_gt_lift)/lossc(x_under_lift,x_gt_lift))
        # loss_ = lossc(image_Rec, image_rec) 
        # --------------------------------------------------------------------------------------------------------------
        # 更新损失并作记录
        train_losses.update(loss_.item(), args.batch_size)  # 整个epoch的平均损失
        train_writer.add_scalar('Train_loss', loss_.item(), n_iter)

        # --------------------------------------------------------------------------------------------------------------
        # 计算梯度并更新参数
        optimizer.zero_grad()
        loss_.backward()
        # 梯度裁剪
        if args.clip_grad_norm:
            clip_grad_norm(model.parameters(), max_norm=10, norm_type=2)
        optimizer.step()

        n_iter += 1

        # --------------------------------------------------------------------------------------------------------------
        # 输出已经处理的数据及相应batch的loss_
        # print(loss_)  # TODO
        if (idx + 1) % 60 == 0:
            logger.info(
                'Epoch-{}-[{}/{}]\t\tLoss: {:.4f}'.format(epoch, (idx + 1) * args.batch_size, total, loss_.item()))
    # ------------------------------------------------------------------------------------------------------------------
    # 返回一个周期的平均损失
    return train_losses.avg


def validate(valid_loader, model, loss, epoch, output_writers, device, logger, args):
    logger.info('\n{}: [{}/{}/X{}] Validation:'.format(args.data_name, args.model, args.modal, args.acceleration))

    loss_ = torch.FloatTensor([0.]).to(device)
    loss_.requires_grad_(True)
    loss_sr = torch.FloatTensor([0.]).to(device)
    loss_sr.requires_grad_(True)
    loss_rec = torch.FloatTensor([0.]).to(device)
    loss_rec.requires_grad_(True)

    valid_losses = AverageMeter()
    valid_psnr = AverageMeter()
    valid_ssim = AverageMeter()
    valid_nrmse = AverageMeter()

    # ------------------------------------------------------------------------------------------------------------------
    # total image numbers
    if args.data_name == 'IXI':
        total = len(Dataloader_Dataset.IXI_dataset(args, 'valid'))
        # print(total)  # TODO
    elif args.data_name == 'fastMRI':
        total = len(Dataloader_Dataset.fastMRI_dataset(args, 'valid'))

    model.eval()

    # ------------------------------------------------------------------------------------------------------------------
    image_recs = []
    image_Recs = []
    image_refs = []
    with torch.no_grad():
        for idx, (name_rec, image_rec, fully, under_sample, mask, name_ref, image_ref, fully_ref) in tqdm(enumerate(valid_loader), desc='Batch',
                                                                  total=len(valid_loader), ncols=80):
            image_rec = image_rec.float().to(device)  # (bs, 1, 256, 256)
            fully = fully.float().to(device)  # (bs, 2, 256, 256)
            under_sample = under_sample.float().to(device)  # (bs, 2, 256, 256)
            mask = mask.float().to(device)  # (bs, 1, 256, 256)
            image_ref = image_ref.float().to(device)
            fully_ref = fully_ref.float().to(device)  # (bs, 2, 256, 256)

            # # --------------------------------------------------------------------------------------------------------------

            # 将under_sample变为复数形式
            under_sample_real = torch.unsqueeze(under_sample[:, 0, :, :], 1)
            under_sample_imag = torch.unsqueeze(under_sample[:, 1, :, :], 1)
            under_sample = torch.complex(under_sample_real, under_sample_imag)  # bs ,1, 256, 256
            under_image_rec = torch.abs(torch.fft.ifft2(under_sample))


            # 参考图片，欠采样图片，欠采样k-space，掩码
            image_Rec, x_rec_lift,x_under_lift,x_gt_lift = model(under_image_rec, under_sample, mask,image_rec)
            loss_ = lossc(image_Rec, image_rec) + 0.1*(lossc(x_rec_lift,x_gt_lift)/lossc(x_under_lift,x_gt_lift))

            valid_losses.update(loss_.item(), image_rec.size(0))  # 整个epoch的平均损失

            # ----------------------------------------------------------------------------------------------------------
            # 计算PSNR，SSIM，NRMSE
            psnr = peak_signal_noise_ratio(np.array(image_rec[0, 0, :, :].data.cpu()),
                                           np.array(image_Rec[0, 0, :, :].clamp(0, 1).data.cpu()),
                                           data_range=np.max(np.array(image_rec[0, 0, :, :].data.cpu())))
            ssim = structural_similarity(np.array(image_rec[0, 0, :, :].data.cpu()),
                                         np.array(image_Rec[0, 0, :, :].clamp(0, 1).data.cpu()))
            nrmse = normalized_root_mse(np.array(image_rec[0, 0, :, :].data.cpu()),
                                        np.array(image_Rec[0, 0, :, :].clamp(0, 1).data.cpu()))
            valid_psnr.update(psnr, image_rec.size(0))
            valid_ssim.update(ssim, image_rec.size(0))
            valid_nrmse.update(nrmse, image_rec.size(0))

            # ----------------------------------------------------------------------------------------------------------
            # 可视化图片质量-周期数的变化
            step = int(total / 8)
            image_idx = [x for x in range(0, 8 * step, step)]
            if idx in image_idx:
                image_recs.append(image_rec)
                image_Recs.append(image_Rec)
                image_refs.append(image_ref)

        image_recs = torch.cat(image_recs, dim=0)  # (8, 1, 256, 256)
        image_Recs = torch.cat(image_Recs, dim=0)
        image_refs = torch.cat(image_refs, dim=0)

        image_recs = vutils.make_grid(image_recs, normalize=True, scale_each=True)  # (3, 256, 2048)
        image_Recs = vutils.make_grid(image_Recs, normalize=True, scale_each=True)
        image_refs = vutils.make_grid(image_refs, normalize=True, scale_each=True)


        output_writers[0].add_image('ground_truth images', torch.unsqueeze(image_recs[0], dim=0), 0)
        output_writers[1].add_image('Rec images ', torch.unsqueeze(image_Recs[0], dim=0), epoch)
        output_writers[2].add_image('Ref images ', torch.unsqueeze(image_refs[0], dim=0), 0)

        # --------------------------------------------------------------------------------------------------------------
        return valid_losses.avg, valid_psnr.avg, valid_ssim.avg, valid_nrmse.avg


def test(test_loader, model, loss, output_writers, device, args):
    print("======================== Start testing! ========================")
    test_psnr = AverageMeter()
    test_ssim = AverageMeter()
    test_nrmse = AverageMeter()
    test_losses = AverageMeter()

    # ------------------------------------------------------------------------------------------------------------------
    # total image numbers
    if args.data_name == 'IXI':
        total = len(Dataloader_Dataset.IXI_dataset(args, 'test'))
    elif args.data_name == 'fastMRI':
        total = len(Dataloader_Dataset.fastMRI_dataset(args, 'test'))

    model.eval()

    # ------------------------------------------------------------------------------------------------------------------
    # /home/shx/My_Projects/My_Net/experiment/IXI/T2/random/MBMSN/X8/test_reconstruction_images
    save_dir = os.path.join(args.root_path, args.save_dir, args.data_name, args.modal,
                            args.mask,
                            args.model,
                            'X{}'.format(args.acceleration), 'test_reconstruction_images')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print("The reconstructed images will be saved in {}".format(save_dir))


    
    save_path = os.path.join(args.root_path, args.save_dir, args.data_name, args.modal,
                         args.mask,
                         args.model,
                         'X{}'.format(args.acceleration))    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    csv_filename = 'instance_metrics.csv'    
    field_names = ['name', 'psnr', 'ssim', 'nrmse']
    with open(os.path.join(save_path, csv_filename), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
        writer.writeheader()

   

    # ------------------------------------------------------------------------------------------------------------------
    image_recs = []
    image_Recs = []
    image_refs = []

    with torch.no_grad():
        # image_rec_128 = None
        for idx, (name_rec, image_rec, fully, under_sample, mask, name_ref, image_ref, fully_ref) in tqdm(enumerate(test_loader), desc='Batch',
                                                                  total=len(test_loader), ncols=80):
            image_rec = image_rec.float().to(device)  # (bs, 1, 256, 256)
            fully = fully.float().to(device)  # (bs, 2, 256, 256)
            under_sample = under_sample.float().to(device)  # (bs, 2, 256, 256)
            mask = mask.float().to(device)  # (bs, 1, 256, 256)
            image_ref = image_ref.float().to(device)
            fully_ref = fully_ref.float().to(device)  # (bs, 2, 256, 256)

            # # --------------------------------------------------------------------------------------------------------------
            # # 将under_sample变为复数形式，并进行数据截断
            # fully_real = torch.unsqueeze(fully[:, 0, :, :], 1)
            # fully_imag = torch.unsqueeze(fully[:, 1, :, :], 1)
            # fully = torch.complex(fully_real, fully_imag)

            # 将under_sample变为复数形式
            under_sample_real = torch.unsqueeze(under_sample[:, 0, :, :], 1)
            under_sample_imag = torch.unsqueeze(under_sample[:, 1, :, :], 1)
            under_sample = torch.complex(under_sample_real, under_sample_imag)  # bs ,1, 256, 256
            under_image_rec = torch.abs(torch.fft.ifft2(under_sample))

            # fully = torch.fft.fftshift(torch.fft.fft2(image_rec))  # [b, 1, H, W]
            # under_sample = fully * mask
            # under_image_rec = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(under_sample)))
            # fully_ref = torch.fft.fftshift(torch.fft.fft2(image_ref))  # [B, 1, H, W]

            # plt.subplot(131), plt.imshow(np.array(image_ref[0, 0, :, :].cpu()), 'gray')
            # plt.subplot(132), plt.imshow(np.array(image_rec[0, 0, :, :].cpu()), 'gray')
            # plt.subplot(133), plt.imshow(np.array(under_image_rec[0, 0, :, :].cpu()), 'gray')
            # plt.show()

            # 参考图片，欠采样图片，欠采样k-space，掩码

            image_Rec = model( under_image_rec, under_sample, mask)

            loss_ = loss(image_Rec, image_rec) 

            test_losses.update(loss_.item(), image_rec.size(0))  # 整个epoch的平均损失

            # ----------------------------------------------------------------------------------------------------------
            # 计算PSNR，SSIM，NRMSE
            psnr = peak_signal_noise_ratio(np.array(image_rec[0, 0, :, :].data.cpu()),
                                           np.array(image_Rec[0, 0, :, :].clamp(0, 1).data.cpu()),
                                           data_range=np.max(np.array(image_rec[0, 0, :, :].data.cpu())))
            ssim = structural_similarity(np.array(image_rec[0, 0, :, :].data.cpu()),
                                         np.array(image_Rec[0, 0, :, :].clamp(0, 1).data.cpu()))
            nrmse = normalized_root_mse(np.array(image_rec[0, 0, :, :].data.cpu()),
                                        np.array(image_Rec[0, 0, :, :].clamp(0, 1).data.cpu()))
            test_psnr.update(psnr, image_rec.size(0))
            test_ssim.update(ssim, image_rec.size(0))
            test_nrmse.update(nrmse, image_rec.size(0))

            # write the metrics into a csv File and visualize
            with open(os.path.join(save_path, csv_filename), 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
                row_data = [name_rec, psnr, ssim, nrmse]
                writer.writerow(dict(zip(field_names, row_data)))

            # ----------------------------------------------------------------------------------------------------------
            # 可视化图片质量-周期数的变化
            step = int(total / 8)
            image_idx = [x for x in range(0, 8 * step, step)]
            if idx in image_idx:
                image_recs.append(image_rec)
                image_Recs.append(image_Rec)
                # image_refs.append(image_ref)

            # ----------------------------------------------------------------------------------------------------------
            # 保存生成的图像数据
            name_rec = name_rec[0].split('.')[0]
            if not os.path.exists(os.path.join(save_dir, 'image_under')):
                os.makedirs(os.path.join(save_dir, 'image_under'), exist_ok=True)
            if not os.path.exists(os.path.join(save_dir, 'image_Rec')):
                os.makedirs(os.path.join(save_dir, 'image_Rec'), exist_ok=True)
            if not os.path.exists(os.path.join(save_dir, 'image_rec')):
                os.makedirs(os.path.join(save_dir, 'image_gt'), exist_ok=True)

            save_path_under = os.path.join(save_dir, 'image_under',  name_rec+'_under.jpg')
            save_path_rec = os.path.join(save_dir, 'image_Rec', name_rec+'_Rec.jpg')
            save_path_gt = os.path.join(save_dir, 'image_gt', name_rec+'_gt.jpg')
            image_Rec = np.array(image_Rec[0, 0, :, :].clamp(0, 1).cpu())
            image_gt = np.array(image_rec[0, 0, :, :].clamp(0, 1).cpu())
            image_under = np.array(under_image_rec[0, 0, :, :].cpu())
            # np.save(save_path, image_sr_256)
            save_image(save_path_under, image_under)
            save_image(save_path_rec, image_Rec)
            save_image(save_path_gt, image_gt)

        image_recs = torch.cat(image_recs, dim=0)  # (8, 1, 256, 256)
        image_Recs = torch.cat(image_Recs, dim=0)


        image_recs = vutils.make_grid(image_recs, normalize=True, scale_each=True)  # (3, 256, 2048)
        image_Recs = vutils.make_grid(image_Recs, normalize=True, scale_each=True)


        output_writers[0].add_image('ground_truth images', torch.unsqueeze(image_recs[0], dim=0), 0)
        output_writers[1].add_image('Rec images ', torch.unsqueeze(image_Recs[0], dim=0), 0)

        return test_losses.avg, test_psnr.avg, test_ssim.avg, test_nrmse.avg
