import argparse
import sys


def get_args():
    parser = argparse.ArgumentParser('EFS-DNN')
    parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. kdd99 or nslkdd)', default='kdd99')
    # parser.add_argument('--tasks', type=str, default="two", choices=["two", "five"], help='two classification or five classification')
    parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=1024, help='The batch size')
    parser.add_argument('--num_in_feat', type=int, default=30, help='The number of input features')
    parser.add_argument('--n_lgb', type=int, default=3, help='The number of LightGBM')
    parser.add_argument('--r_sample', type=float, default=0.5, help='The ratio of sampled instances')
    parser.add_argument('--classes', type=int, default=2, help='The number of classes')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=0, help='Random Seed')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    return args

