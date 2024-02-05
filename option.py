# -- coding: utf-8 --
import argparse
import yaml


def get_parser():
    parser = argparse.ArgumentParser(description='Main function arguments')

    # Hardware specifications
    parser.add_argument('--n_threads', type=int, default=6,
                        help='number of threads for data loading')
    parser.add_argument('--cpu', action='store_true',
                        help='use cpu only')
    parser.add_argument('--n_GPUs', type=int, default=1,
                        help='number of GPUs')
    parser.add_argument('--gpuid', type=str, default='7',
                        help='GPU ID')
    parser.add_argument('--seed', type=int, default=6,
                        help='random seed')

    # Data specifications
    parser.add_argument('--data_dir', type=str, default='/home/shx/My_Projects/datasets',
                        help='Datasets root directory')
    parser.add_argument('--data_name', type=str, default='BraTS',
                        help='Data name: IXI, fastMRI')
    parser.add_argument('--modal', type=str, default='T2',
                        help='the modal of data')
    parser.add_argument('--acceleration', type=int, default=4,
                        help='acceleration: 2, 4, 6, 8')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size')
    parser.add_argument('--mask', type=str, default='random',
                        help='random, equispaced_fraction')

    # Model specifications
    parser.add_argument('--upscale_factor', type=int, default=2,
                        help='upscale factor:2, 4, 6, 8')
    parser.add_argument('--input_type', type=str, default='k_data',
                        help='k_data, image')
    parser.add_argument('--model', type=str, default='unet',
                        help='model type')
    parser.add_argument('--root_path', type=str, default='/home/shx/My_Projects/My_Net' ,
                        help='root path')
    parser.add_argument('--n_blocks', type=int, default=5,
                        help='the number of blocks')

    # Training specifications
    parser.add_argument('--train', type=str, default='train',
                        help='train or test')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='training epochs')
    parser.add_argument('--resume', action='store_true',
                        help='whether to continue train')
    parser.add_argument('--save_dir', type=str, default='experiment',
                        help='directory to save results ')
    parser.add_argument('--loss', type=str, default='l1',
                        help='loss function')
    parser.add_argument('--clip_grad_norm', '-cgn', action="store_true",
                        help="clip gradient and norm")

    # Optimization specifications
    ## 学习率的设置和衰减方式
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--lr_policy', type=str, default='step',
                        help='learning rate decay mode:steplr/multisteplr/lambdalr')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='learning rate decay factor')
    parser.add_argument('--step_size', type=int, default=10,
                        help='StepLR parameter')
    parser.add_argument('--multistep_size', type=int, default=15,
                        help='MultiStepLR parameter')
    parser.add_argument('--epoch_decay', type=int, default=20,
                        help='lambda parameter, the number of iterations that begin to decay')
    ## 优化器的选择
    parser.add_argument('--optimizer', default='Adam', choices=('SGD', 'Adam', 'RMSprop'),
                        help='optimizer to use (SGD | Adam | RMSprop)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (L2 penalty)')
    return parser


parser = get_parser()

with open(r"F:\paper\my paper\2023_ac3m\code\Mc_Rec_Net_cjc\config.yaml", 'r', encoding='utf-8') as f:
    default_args = yaml.load(f, Loader=yaml.FullLoader)

# with open("/export/home/lewie/shx/Test_Network/config.yaml", 'r', encoding='utf-8') as f:
#     default_args = yaml.load(f, Loader=yaml.FullLoader)

parser.set_defaults(**default_args)
args = parser.parse_args()

if __name__ == '__main__':
    print(args)
    # print()
    for i in args.slice_indexes:
        print(type(i))
    # a = args.list1
    # print(a.split(','))
