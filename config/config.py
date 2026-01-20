import argparse

alpha = 0.9
decay = 0.0002801
drop_all = 0.2
drop_cell = 0.1
epoch = 700
gamma = 0.99
hid_dim1 = 256
hid_dim2 = 512
layer_drug = 3
layer_cell = 3

ln_cell = False
ln_drug = True
lr = 0.0001
output_dim_all = 3
output_dim_cell = 1024
output_dim_drug = 256
pat = 30
seed = 694
share = False
sqrt_norm = True
tau = 0.5535
beta = 0.6163

gpu = 0


def parse():
    parser = argparse.ArgumentParser(description='Cell_Drug_Response_pre')

    parser.add_argument('--seed', dest='seed', type=int, default=seed, help='')
    parser.add_argument('--epoch', dest='epoch', type=int, default=epoch, help='')
    parser.add_argument('--gamma', dest='gamma', type=float, default=gamma, help='')
    parser.add_argument('--pat', dest='pat', type=int, default=pat, help='')

    parser.add_argument('--lr', dest='lr', type=float, default=lr, help='')
    parser.add_argument('--decay', dest='decay', type=float, default=decay, help='')

    parser.add_argument('--alpha', dest='alpha', type=float, default=alpha, help='')

    parser.add_argument('--drop_all', dest='drop_all', type=float, default=drop_all, help='')
    parser.add_argument('--drop_cell', dest='drop_cell', type=float, default=drop_cell, help='')

    parser.add_argument('--layer_drug', dest='layer_drug', type=int, default=layer_drug, help='')
    parser.add_argument('--layer_cell', type=int, default=2)


    parser.add_argument('--output_dim_drug', dest='output_dim_drug', type=int, default=output_dim_drug, help='')
    parser.add_argument('--output_dim_cell', dest='output_dim_cell', type=int, default=output_dim_cell, help='')
    parser.add_argument('--output_dim_all', dest='output_dim_all', type=int, default=output_dim_all, help='')
    parser.add_argument('--hid_dim1', dest='hid_dim1', type=int, default=hid_dim1, help='')
    parser.add_argument('--hid_dim2', dest='hid_dim2', type=int, default=hid_dim2, help='')

    parser.add_argument('--share', dest='share', type=bool, default=share, help='')
    parser.add_argument('--ln_drug', dest='ln_drug', type=bool, default=ln_drug, help='')
    parser.add_argument('--ln_cell', dest='ln_cell', type=bool, default=ln_cell, help='')
    parser.add_argument('--sqrt_norm', dest='sqrt_norm', type=bool, default=sqrt_norm, help='')

    parser.add_argument('--beta', dest='beta', type=float, default=beta, help='')
    parser.add_argument('--tau', dest='tau', type=float, default=tau, help='')

    parser.add_argument('--gpu', dest='gpu', type=int, default=gpu, help='')

    args = parser.parse_args()

    return args
