import argparse
import sys
import time
import os


def parse_args():
    """ this class includes training options """

    # basic parameters
    parser = argparse.ArgumentParser(description='DeepLatteTest')

    # load data from file
    parser.add_argument('--data_path', type=str, default='./sample_data/los_angeles_500m_2020_02.npz', help='data path')
    parser.add_argument('--result_dir', type=str, default='./sample_data/results/', help='result directory')
    parser.add_argument('--model_dir', type=str, default='./sample_data/results/', help='model directory')
    parser.add_argument('--model_name', type=str, default='los_angeles_500m_2020_03', help='model name')

    # training parameters
    parser.add_argument('--device', type=str, default='3', help='GPU id')

    # model parameters
    parser.add_argument('--seq_len', type=int, default=6, help='sequence length for rnn')
    parser.add_argument('--en_features', type=str, default='64,16', help='encoder sizes')
    parser.add_argument('--de_features', type=str, default='16,64', help='decoder sizes')
    parser.add_argument('--kernel_sizes', type=str, default='1,3,5', help='kernel sizes for convolution operation')
    parser.add_argument('--h_channels', type=int, default=32, help='number of channels for convolution operation')
    parser.add_argument('--fc_h_features', type=int, default=32, help='hidden size for the fully connected layer')

    args = parser.parse_args()
    input_check(args)
    return args


def input_check(args):

    # check data path existence
    if not os.path.exists(args.data_path):
        print(f'The data path does not exist: {args.data_path}.')
        sys.exit(-1)

    # check result path existence
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # check model path existence
    model_file = os.path.join(args.model_dir, args.model_name + '.pkl')
    if not os.path.exists(model_file):
        print('Model file does not exist.')
        sys.exit(-1)

