import torch
import os
import shutil


def save_checkpoint(model, optimizer, lr_schedule, ckpt_dir, filename, is_best, metrics, epoch=0):
    checkpoint = {
        'models': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_schedule': lr_schedule.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, os.path.join(ckpt_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(ckpt_dir, filename), os.path.join(ckpt_dir, 'ckpt_best.pth'))

def save_checkpoint_adv(model, optimizer, optimizer_adv, lr_schedule, lr_schedule_adv, ckpt_dir, filename, is_best, metrics, epoch=0):
    checkpoint = {
        'models': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'optimizer_adv': optimizer_adv.state_dict(),
        'lr_schedule': lr_schedule.state_dict(),
        'lr_schedule_adv': lr_schedule_adv.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, os.path.join(ckpt_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(ckpt_dir, filename), os.path.join(ckpt_dir, 'ckpt_best.pth'))