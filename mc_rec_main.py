import os
import datetime
import time

from torch.utils.tensorboard import SummaryWriter

import torch
import csv
import torch.nn as nn

from option import args
from data.Mc_Rec_Dataloader_Dataset import Loader
from utils.seed import set_random_seed
from utils.config import save_config
from utils.logger import get_logger
from utils.boundary import generate_boundary
from utils.optimizer import get_optimizer
from utils.scheduler import get_lr_scheduler
from utils.save_checkpoint import save_checkpoint
from utils.visualize_metrics import visualize

# 导入实验用的模型
from models import Model

from loss.specify_loss import specify_loss

# from mc_rec_train_and_test import train, validate, test
from mc_rec_train_and_test_ac3m import train, validate, test
# from mc_rec_train_and_test_puert2 import train, validate, test
# from mc_rec_train_and_test_dudornet import train, validate, test
# from train_and_test_t2net import train, validate, test
import sys

sys.path.append('models')

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

    # ------------------------------------------------------------------------------------------------------------------
    # set random seed
    print('==> Set random seed......')
    set_random_seed(args.seed)
    print('\tFinish!\n')

    # ------------------------------------------------------------------------------------------------------------------
    # save models and results in save_path（保存模型输出结果的目录）
    save_path = os.path.join(args.root_path, args.save_dir, args.data_name, args.modal,
                             args.mask,
                             args.model,
                             'X{}'.format(args.acceleration))
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    print('==> Will save everything to {}\n'.format(save_path))

    # ------------------------------------------------------------------------------------------------------------------
    # 保存config.txt
    print('==> Start saving configure......')
    now = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
    config_dir = os.path.join(save_path, 'config')
    if not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)
    open_type_c = 'w'
    config_path = os.path.join(config_dir, '{}_{}_config.txt'.format(args.train, now))
    save_config(config_path, open_type_c, now, args)
    print('\tFinish!\n')

    # ------------------------------------------------------------------------------------------------------------------
    # 自己定义的函数模型
    model_ = Model(args)
    model = model_.model
    

    if args.n_GPUs > 1:
        model = nn.DataParallel(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('\t{} is ready!\n'.format(args.model.upper()))

    # ------------------------------------------------------------------------------------------------------------------
    # Specify loss function（确定损失函数）
    loss = specify_loss(args).to(device)

    # ------------------------------------------------------------------------------------------------------------------
    # 训练
    if args.train == 'train':

        # --------------------------------------------------------------------------------------------------------------
        # save training data in train and val folder for visualization（保存实验过程中的数据，用于tensorboardx的可视化处理）
        train_writer = SummaryWriter(os.path.join(save_path, 'tensorboardx_train'))
        valid_writer = SummaryWriter(os.path.join(save_path, 'tensorboardx_valid'))
        output_writers = []

        # --------------------------------------------------------------------------------------------------------------
        # 分别用于存放ground_truth, rec_image, under_sample
        for i in ['ground_truth', 'Rec_image', 'Ref_image']:
            output_writers.append(SummaryWriter(os.path.join(save_path, 'tensorboardx_valid', '{}'.format(i))))

        # --------------------------------------------------------------------------------------------------------------
        # initialize input data（初始化模型的输入数据形式）
        print('==> Start processing dataloader......')
        loader = Loader(args)
        train_loader = loader.trainloader()
        valid_loader = loader.validloader()
        print('\tFinish!\n')

        # --------------------------------------------------------------------------------------------------------------
        # 初始化优化器和学习率衰减的方法
        print("==> Preparing optimizer / lr_schedule......")
        optimizer = get_optimizer(args, model)
        lr_schedule = get_lr_scheduler(optimizer, args)
        print('\tFinish!\n')

        # --------------------------------------------------------------------------------------------------------------
        # log.txt保存的路径及其打开方式的初始化
        log_dir = os.path.join(save_path, 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, '{}_log.txt'.format(args.train))
        open_type_l = 'w'

        # --------------------------------------------------------------------------------------------------------------
        # best metrics initialize
        start_epoch = 0
        best_epoch = 1  # best_loss对应的epoch
        best_psnr = 0  # best_loss对应的psnr
        best_ssim = 0  # best_loss对应的ssim
        best_nrmse = 0  # best_loss对应的nrmse

        # --------------------------------------------------------------------------------------------------------------
        # 判断是否为resume
        if args.resume:
            open_type_l = 'a'
            generate_boundary(log_path, open_type_l, now)

            print('==> Starting loading state_dict......')
            ckpt_path = os.path.join(save_path, 'checkpoint', 'ckpt_latest.pth')
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['models'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['lr_schedule']['last_epoch']
            best_psnr = checkpoint['metrics']['best_psnr']
            best_epoch = checkpoint['metrics']['best_epoch']
            best_ssim = checkpoint['metrics']['best_ssim']
            print('\tFinish!\n')
        else:
            generate_boundary(log_path, open_type_l, now)

        # --------------------------------------------------------------------------------------------------------------
        # create CSV File to store metrics
        csv_filename = 'metrics.csv'
        field_names = ['epoch', 'loss', 'psnr', 'ssim', 'nrmse']
        with open(os.path.join(save_path, csv_filename), open_type_l) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
            writer.writeheader()

        # --------------------------------------------------------------------------------------------------------------
        # longger用于控制log.txt文件的输入以及控制台的输出
        logger = get_logger(log_path)
        logger.info('======================== Start training! ========================\n')

        # --------------------------------------------------------------------------------------------------------------
        # 若为中断继续训练，需更新lr
        for i in range(start_epoch):
            optimizer.step()
            lr_schedule.step()

        for epoch in range(start_epoch + 1, args.n_epochs + 1):
            start = time.time()

            # 训练集
            train_loss = train(train_loader, model, loss, optimizer, epoch, train_writer, device, logger, args)
            train_writer.add_scalar('Average Train_Loss', train_loss, epoch)

            # 验证集
            valid_loss, psnr, ssim, nrmse = validate(valid_loader, model, loss, epoch, output_writers, device, logger,
                                                     args)
            valid_writer.add_scalar('Average Valid_loss', valid_loss, epoch)
            valid_writer.add_scalar('Average PSNR', psnr, epoch)
            valid_writer.add_scalar('Average SSIM', ssim, epoch)
            valid_writer.add_scalar('Average NRMSE', nrmse, epoch)

            # 更新学习率
            lr_schedule.step()

            # write the metrics into a csv File and visualize
            with open(os.path.join(save_path, csv_filename), 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
                row_data = [epoch, valid_loss, psnr, ssim, nrmse]
                writer.writerow(dict(zip(field_names, row_data)))

            # visualize(save_path, csv_filename)
            # 判断是否是最优的PSNR，如果是，则更新loss/psnr/ssim/nrmse并输出
            if best_psnr < 0:
                best_psnr = psnr

            if epoch == 1:
                is_best = psnr >= best_psnr
            else:
                is_best = psnr > best_psnr
            if is_best:
                best_epoch = epoch
                best_ssim = ssim
                # best_nrmse = nrmse
            best_psnr = max(psnr, best_psnr)
            metrics = {
                'best_psnr': best_psnr,
                'best_ssim': best_ssim,
                'best_epoch': best_epoch
            }

            logger.info(
                'Epoch: [{}]\tValid Loss: {:.4f}\tPSNR: {:.3f}\tSSIM: {:.3f}\t(Best PSNR: {:.3f}\tSSIM: {:.3f}\t@Epoch: {})\n'
                    .format(epoch, valid_loss, psnr, ssim, best_psnr, best_ssim, best_epoch))

            # 保存模型/优化器/学习率衰减器的参数
            logger.info('Saving models...')
            ckpt_dir = os.path.join(save_path, 'checkpoint')
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)

            # 保存最后一次的参数
            filename = 'ckpt_latest.pth'
            save_checkpoint(model, optimizer, lr_schedule, ckpt_dir, filename, is_best, metrics)

            # 输出训练一个周期所需要的时间
            end = time.time()
            time_consume = end - start
            logger.info('Training consuming: {:.2f}s'.format(time_consume))
            logger.info('==================================================\n')

        # --------------------------------------------------------------------------------------------------------------
        # 输出Finish training!
        logger.info('========================= Finish training! =========================')

    elif args.train == 'test':

        # --------------------------------------------------------------------------------------------------------------
        # initialize input data（初始化模型的输入数据形式）
        print('==> Start loading testloader......')
        loader = Loader(args)
        test_loader = loader.testloader()
        print('\tFinish!\n')

        # --------------------------------------------------------------------------------------------------------------
        # 加载模型参数
        print('==> Starting loading state_dict......')
        ckpt_path = os.path.join(save_path, 'checkpoint', 'ckpt_best.pth')
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['models'],strict=False)
        print('\tFinish!\n')

        # --------------------------------------------------------------------------------------------------------------
        # test_writer = SummaryWriter(os.path.join(save_path, 'tensorboardx_test'))
        output_writers = []
        for i in ['ground_truth', 'Rec_image', 'ref_image']:
            output_writers.append(
                SummaryWriter(os.path.join(save_path, 'tensorboardx_test', '{}'.format(i))))

        test_loss, psnr, ssim, nrmse = test(test_loader, model, loss, output_writers, device, args)

        print('========================= Finish testing! =========================\n')

        # --------------------------------------------------------------------------------------------------------------
        # 记录测试指标结果
        test_result_path = os.path.join(save_path, 'metric_results.txt')
        generate_boundary(test_result_path, 'w', now)
        open_type = 'a'
        logger = get_logger(test_result_path, open_type)
        logger.info("Test Loss: {:.4f}\tPSNR: {:.3f}\tSSIM: {:.3f}\tNRMSE: {:.3f}".format(test_loss, psnr, ssim, nrmse))


    else:
        return NotImplementedError("[%s] is not implemented", args.train)


if __name__ == '__main__':
    main_start = time.time()
    main()
    main_end = time.time()
    print('Total consuming: {:.2f}s'.format(main_end - main_start))
