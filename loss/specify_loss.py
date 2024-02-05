from loss.percentual import Percentual
from loss.char_edge import Sum
import torch.nn as nn


def specify_loss(args):
    if args.loss == 'l1':
        loss = nn.L1Loss()

    if args.loss == 'l2':
        loss = nn.MSELoss()

    if args.loss == 'percentual':
        loss = Percentual()

    if args.loss == 'char_edge':
        loss = Sum()
    return loss
