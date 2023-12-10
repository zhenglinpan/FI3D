import torch
from options.options_vae import Options

from utils.utils import *
import utils.paramUtil as paramUtil
from utils.plot_script import plot_3d_motion
from scripts.motion_process import recover_from_ric
from os.path import join as pjoin


def plot_t2m(index, motion, mean, std, save_dir, motion_type, perturb='none'):   
    motion = motion * std + mean 
    
    joint = recover_from_ric(torch.from_numpy(motion).float(), 22).numpy()
    joint = motion_temporal_filter(joint, sigma=1)
    save_name = f'{index}_{motion_type}_{perturb}'
    plot_3d_motion(os.path.join(save_dir, save_name + '.mp4'), paramUtil.t2m_kinematic_chain, joint, title=motion_type, fps=20)

if __name__=="__main__":
    parser = Options()
    opt = parser.parse()
    
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)
    
    joint_dir = './data_fi3d/joint'
    flist = os.listdir(joint_dir)
    perturbs = [f.strip('.npy').split('_')[-2] + '_' + f.strip('.npy').split('_')[-1] for f in flist if 'joint_gen' in f]
    
    perturbs = [f for f in perturbs if 'quater' in f]
    
    mean, std = np.load(pjoin(opt.meta_dir, 'mean.npy')), np.load(pjoin(opt.meta_dir, 'std.npy'))
    gt = np.load('./data_fi3d/joint/joint_gt.npy')
    gen = np.load('./data_fi3d/joint/joint_gen_none_None.npy')
    
    save_dir = './data_fi3d/animations'
    for i in range(2):
        plot_t2m(i, gt[i], mean, std, save_dir, motion_type='gt')
        plot_t2m(i, gen[i], mean, std, save_dir, motion_type='gen')
        
        for perturb in perturbs:
            print(f'Animating {perturb} motions')
            genp = np.load(f'./data_fi3d/joint/joint_gen_{perturb}.npy')
            plot_t2m(i, genp[i], mean, std, save_dir, motion_type='genp', perturb=perturb)
