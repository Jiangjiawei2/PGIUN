import torch.optim as optim
import itertools

def get_optimizer(args, model):
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

    elif args.optimizer == 'Adam':
        # optimizer = optim.Adam(model.parameters(),
        #                        lr=args.lr,
        #                        weight_decay=args.weight_decay)
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               betas=(0.5, 0.999),
                               weight_decay=args.weight_decay)                      

    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)


    else:
        return NotImplementedError("Optimizer [%s] is not implemented", args.optimizer)

    return optimizer

def get_optimizer_for_adversarial(args, model):
    netD_A = model.get_discriminator_A()
    netD_A_edge = model.get_discriminator_A_edge()

    if args.optimizer == 'Adam':
        optimizer_adv = optim.Adam(
            itertools.chain(netD_A.parameters(), netD_A_edge.parameters()),
            lr=args.lr,
            betas=(0.5, 0.999),
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'SGD':
        optimizer_adv = optim.SGD(
            itertools.chain(netD_A.parameters(), netD_A_edge.parameters()),
            lr=args.lr,
            momentum=args.momentum_adv,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'RMSprop':
        optimizer_adv = optim.RMSprop(
            itertools.chain(netD_A.parameters(), netD_A_edge.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise NotImplementedError("Optimizer [%s] is not implemented" % args.optimizer)

    return optimizer_adv
