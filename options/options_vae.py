# from options.base_options import BaseOptions
import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--name', type=str, default="Decomp_SP001_SM001_H512", help='Name of this trial')
        self.parser.add_argument("--gpu_id", type=int, default=0, help='GPU id')
        self.parser.add_argument('--dataset_name', type=str, default='t2m', help='Dataset Name')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument("--window_size", type=int, default=24, help="Length of motion clips for reconstruction")
        self.parser.add_argument('--dim_movement_enc_hidden', type=int, default=512, help='Dimension of hidden in AutoEncoder(encoder)')
        self.parser.add_argument('--dim_movement_dec_hidden', type=int, default=512, help='Dimension of hidden in AutoEncoder(decoder)')
        self.parser.add_argument('--dim_movement_latent', type=int, default=512, help='Dimension of motion snippet')
        self.parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
        self.parser.add_argument('--max_epoch', type=int, default=270, help='Training iterations')
        self.parser.add_argument('--feat_bias', type=float, default=5, help='Layers of GRU')
        self.parser.add_argument('--lambda_sparsity', type=float, default=0.001, help='Layers of GRU')
        self.parser.add_argument('--lambda_smooth', type=float, default=0.001, help='Layers of GRU')
        self.parser.add_argument('--lambda_normality', type=float, default=0.001, help='Layers of GRU')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='Layers of GRU')
        self.parser.add_argument('--is_continue', action="store_true", help='Training iterations')
        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress')
        self.parser.add_argument('--save_every_e', type=int, default=10, help='Frequency of printing training progress')
        self.parser.add_argument('--eval_every_e', type=int, default=3, help='Frequency of printing training progress')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of printing training progress')
        self.parser.add_argument("--unit_length", type=int, default=4, help="Length of motion")
        
    def parse(self):
        self.opt = self.parser.parse_args()
        self.opt.is_train = True
        args = vars(self.opt)
        return self.opt