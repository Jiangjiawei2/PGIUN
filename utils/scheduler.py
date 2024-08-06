from torch.optim import lr_scheduler
from warmup_scheduler import GradualWarmupScheduler

"""
lr的更新方法
"""


def get_lr_scheduler(optimizer, args, last_epoch=-1):
    if 'lr_policy' not in args or args.lr_policy == 'constant':
        scheduler = None

    elif args.lr_policy == 'step':
        lr_schedule = lr_scheduler.StepLR(optimizer,
                                          step_size=args.step_size,
                                          gamma=args.gamma,
                                          last_epoch=last_epoch)
    elif args.lr_policy == 'cosine':
        lr_schedule = lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs, eta_min=1e-6)

    elif args.lr_policy == 'multistep':
        milestones = [x for x in range(args.multistep_size, args.n_epochs, args.multistep_size)]
        lr_schedule = lr_scheduler.MultiStepLR(optimizer,
                                               milestones=milestones,
                                               gamma=args.gamma,
                                               last_epoch=last_epoch)

    elif args.lr_policy == 'lambda':
        def lambda_rule(ep):
            """
            学习率衰减因子的变化函数
            :param ep: 训练周期数迭代次数
            """
            lr_1 = 1.0 - max(0, ep - args.epoch_decay) / float(args.n_epochs - args.epoch_decay + 1)
            return lr_1

        lr_schedule = lr_scheduler.LambdaLR(optimizer,
                                            lr_lambda=lambda_rule,
                                            last_epoch=last_epoch)

    elif args.lr_policy == 'warmup':
        warmup_epochs = 5
        scheduler_cosine = lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs - warmup_epochs, eta_min=1e-6)
        lr_schedule = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                           after_scheduler=scheduler_cosine)


    else:
        return NotImplementedError("learning rate policy [%s] is not implemented", args.lr_policy)

    return lr_schedule

