import os
from os.path import join as pjoin

import torch
import numpy as np
from torch.utils.data import DataLoader

from options.options_vae import Options
from networks.modules_vae import MovementEncoder, MovementDecoder
from networks.trainer_fi3d_eval import TrainerMotionVAE
from data.dataset_vae_eval import MotionDataset

from networks.modules import MotionEncoderBiGRUCo

from utils.utils import *
import utils.paramUtil as paramUtil
from utils.plot_script import plot_3d_motion
from scripts.motion_process import recover_from_ric

def plot_t2m(motion, mean, std, save_dir, motion_type='gt'):
    assert motion_type in ['gt', 'gen']
    motion = motion * std + mean
    for i, joint_data in enumerate(motion):
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = '%s_%02d_%s'%(save_dir, i, motion_type)
        joint = motion_temporal_filter(joint, sigma=1)
        np.save(save_path + '_' + motion_type + '.npy', joint)
        plot_3d_motion(save_path + '_' + motion_type + '.mp4', paramUtil.t2m_kinematic_chain, joint, title=motion_type, fps=20)

if __name__ == '__main__':
    parser = Options()
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        # self.opt.gpu_id = int(self.opt.gpu_id)
        torch.cuda.set_device(opt.gpu_id)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        # opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.max_motion_length = 196
        dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        # opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.kit_kinematic_chain
    else:
        raise KeyError('Dataset Does Not Exist')

    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    # w_vectorizer = WordVectorizer('./glove', 'our_vab')
    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    movement_enc = MovementEncoder(dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)    # (247, 512, 512) for KIT and (259, 512, 512) for T2M
    movement_dec = MovementDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, dim_pose)    # (512, 512, 251) for KIT and (512, 512, 263) for T2M
    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'text_mot_match', 'model', 'finest.tar'), map_location=opt.device)
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    # print(checkpoint.keys())
    
    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent, hidden_size=1024, output_size=512, device=opt.device)
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    
    all_params = 0
    pc_mov_enc = sum(param.numel() for param in movement_enc.parameters())
    print(movement_enc)
    print("Total parameters of motion encoder (prior net): {}".format(pc_mov_enc))   # 1,842,688
    all_params += pc_mov_enc

    pc_mov_dec = sum(param.numel() for param in movement_dec.parameters())
    print(movement_dec)
    print("Total parameters of motion decoder (posterior net): {}".format(pc_mov_dec))  # 1,657,407
    all_params += pc_mov_dec

    trainer = TrainerMotionVAE(opt, movement_enc, movement_dec, motion_enc)

    # the returns of the datasets were frames number of length 538k and 536k
    gt_dataset = MotionDataset('./eval_data/gt', opt, mean, std, train_split_file)
    gen_dataset = MotionDataset('./eval_data/gen', opt, mean, std, val_split_file)

    print('gt dataset:', len(gt_dataset), 'gen dataset:', len(gen_dataset))

    gt_loader = DataLoader(gt_dataset, batch_size=128, drop_last=True, num_workers=4, shuffle=False, pin_memory=True)
    gen_loader = DataLoader(gen_dataset, batch_size=128, drop_last=True, num_workers=4, shuffle=False, pin_memory=True)

    trainer.eval(gt_loader, gen_loader)