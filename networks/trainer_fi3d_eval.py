import os, time
import math
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from collections import OrderedDict
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from networks.modules import MovementConvEncoder, TextEncoderBiGRUCo, MotionEncoderBiGRUCo

from utils.word_vectorizer import POS_enumerator

class Logger(object):
  def __init__(self, log_dir):
    pass

  def scalar_summary(self, tag, value, step):
    pass


def print_current_loss(start_time, niter_state, total_niters, losses, epoch=None, inner_iter=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    print('epoch: %03d inner_iter: %5d' % (epoch, inner_iter), end=" ")
    # now = time.time()
    message = '%s niter: %07d completed: %3d%%)'%(time_since(start_time, niter_state / total_niters), niter_state, niter_state / total_niters * 100)
    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)
    print(message)


class TrainerMotionVAE(object):
    def __init__(self, args, movement_enc, movement_dec, motion_enc):
        self.opt = args
        self.movement_enc = movement_enc
        self.movement_dec = movement_dec
        self.motion_encoder = motion_enc
        self.device = args.device
        
        if args.is_train:
            self.logger = Logger(args.log_dir)
            self.sml1_criterion = torch.nn.SmoothL1Loss()
            self.l1_criterion = torch.nn.L1Loss()
            self.mse_criterion = torch.nn.MSELoss()
    
    
    @staticmethod
    def kl_criterion(mu1, logvar1, mu2, logvar2):
        # KL( N(mu1, sigma2_1) || N(mu_2, sigma2_2))
        # loss = log(sigma2/sigma1) + (sigma1^2 + (mu1 - mu2)^2)/(2*sigma2^2) - 1/2
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
        return kld.sum() / mu1.shape[0]
    
    @staticmethod
    def kl2norm_criterion(p, dim=1):
        '''
        calcualte kl divergence between 3-D embeddings and its hypothetical normal distribution
        
        '''
        mu = torch.mean(p, dim=dim, keepdim=True)   
        sig = torch.std(p, dim=dim, keepdim=True)
        print(mu.shape, sig.shape)  # should be torch.Size([b, 1, 512])
        
        norm = torch.normal(mu, sig)
        print(norm.shape)   # should be torch.Size([b, c, 512])
        
        softmax = torch.nn.Softmax(dim=1)
        
        p_dis = softmax(p)
        norm_dis = softmax(norm)

        kld = torch.sum(p_dis * torch.log(p_dis / norm_dis), dim=dim)
        print(kld.shape)       # should be torch.Size([b, 1, 512])

        return kld
        

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def forward(self, batch_data, m_lens):
        joints = batch_data
        
        joints = joints.detach().to(self.device).float()     
        latents = self.movement_enc(joints[..., :-4])      # take out the last 4 columns, which are root positions torch.Size([128, 24, 259])

        motion_embedding = self.motion_encoder(latents, m_lens)
        
        return motion_embedding.detach().cpu().numpy()
        
    def backward(self):
        self.loss_rec = self.l1_criterion(self.recon_motions, self.motions)
        self.loss_sparsity = torch.mean(torch.abs(self.latents))
        self.loss_smooth = self.l1_criterion(self.latents[:, 1:], self.latents[:, :-1])
        # self.loss_normality = self.kl2norm_criterion(self.latents)
        
        self.loss = self.loss_rec + self.loss_sparsity * self.opt.lambda_sparsity +\
                    self.loss_smooth*self.opt.lambda_smooth
    
    def update(self):
        self.zero_grad([self.opt_movement_enc, self.opt_movement_dec])
        self.backward()
        self.loss.backward()
        self.step([self.opt_movement_enc, self.opt_movement_dec])

        loss_logs = OrderedDict({})
        loss_logs['loss'] = self.loss_rec.item()
        loss_logs['loss_rec'] = self.loss_rec.item()
        loss_logs['loss_sparsity'] = self.loss_sparsity.item()
        loss_logs['loss_smooth'] = self.loss_smooth.item()
        # loss_logs['loss_normality'] = self.loss_normality.item()

        return loss_logs

    def save(self, file_name, ep, total_it):
        state = {
            'movement_enc': self.movement_enc.state_dict(),
            'movement_dec': self.movement_dec.state_dict(),

            'opt_movement_enc': self.opt_movement_enc.state_dict(),
            'opt_movement_dec': self.opt_movement_dec.state_dict(),

            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)
        return

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)

        self.movement_dec.load_state_dict(checkpoint['movement_dec'])
        self.movement_enc.load_state_dict(checkpoint['movement_enc'])

        self.opt_movement_enc.load_state_dict(checkpoint['opt_movement_enc'])
        self.opt_movement_dec.load_state_dict(checkpoint['opt_movement_dec'])

        return checkpoint['ep'], checkpoint['total_it']

    def eval(self, gt_dataloader, gen_dataloader):
        self.movement_enc.to(self.device)
        self.movement_dec.to(self.device)
        self.motion_encoder.to(self.device)

        self.opt_movement_enc = optim.Adam(self.movement_enc.parameters(), lr=self.opt.lr)
        self.opt_movement_dec = optim.Adam(self.movement_dec.parameters(), lr=self.opt.lr)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(gt_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(gt_dataloader), len(gen_dataloader)))
        val_loss = 0
        logs = OrderedDict()
        
        self.movement_dec.eval()
        self.movement_enc.eval()
        
        all_motion_gt = []
        print("processing gt data")
        print("len(gt_dataloader)", len(gt_dataloader))
        for i, (batch_data, m_lens) in enumerate(gt_dataloader):  # torch.Size([128, 196, 263])
            m_lens = m_lens // self.opt.unit_length     # self.opt.unit_length = 4 by default
            batch_data = batch_data.detach().to(self.device).float()
            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()   # sort by length, descending
            batch_data = batch_data[align_idx]   
            m_lens = m_lens[align_idx]
            
            motion = self.forward(batch_data, m_lens)   # ([128, 512])  # movement: torch.Size([128, 49, 512])
            motion = np.expand_dims(motion, axis=1)     # ([128, 1, 512])
            
            all_motion_gt.append(motion)
            all_gt = batch_data if i == 0 else torch.cat([all_gt, batch_data], dim=0)
            
        all_motion_gt = np.array(all_motion_gt)
        all_motion_gt = np.reshape(all_motion_gt, (all_motion_gt.shape[0]*all_motion_gt.shape[1], all_motion_gt.shape[2], all_motion_gt.shape[3]))
        # print("gt all_movement", all_movement.shape)  # (4608, 49, 512)
        
        np.save(f'/data_fi3d/emb/emb_gt.npy', all_motion_gt)
        np.save(f'./data_fi3d/joint/joint_gt.npy', all_gt.detach().cpu().numpy())
        
        augmentations = {
                    'none':    [None], 
                    'uniform':      [1e-3, 1e-2, 1e-1, 1], 
                    'gaussian':     [1e-3, 1e-2, 1e-1, 1], 
                    'smoothing':    [(1, 1), (3, 1), (5, 1), (10, 1)], 
                    'skipping':     [5, 10, 30, 50], 
                    'shuffling':    [5, 10, 30, 50], 
                    'repeating':    ['half', 'quater'],
                    }
        for key, values in augmentations.items():
            print("processing gen data")
            for value in values:
                print(f"using {key}, {value} to augment data")
                all_motion_gen = []
                for i, (batch_data, m_lens) in enumerate(gen_dataloader):
                    m_lens = m_lens // self.opt.unit_length     # self.opt.unit_length = 4 by default
                    batch_data = batch_data.detach().to(self.device).float()
                    align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()   # sort by length, descending
                    batch_data = batch_data[align_idx]   
                    m_lens = m_lens[align_idx]
                    
                    batch_data = self.augment(batch_data, method=key, value=value)
                    
                    motion = self.forward(batch_data, m_lens)   
                    motion = np.expand_dims(motion, axis=1)     # ([128, 1, 512])

                    all_motion_gen.append(motion)
                    all_gen = batch_data if i == 0 else torch.cat([all_gen, batch_data], dim=0)
                    
                all_motion_gen = np.array(all_motion_gen)
                all_motion_gen = np.reshape(all_motion_gen, (all_motion_gen.shape[0]*all_motion_gen.shape[1], all_motion_gen.shape[2], all_motion_gen.shape[3]))
                # print("gen all_movement", all_movement.shape) # (4608, 49, 512)
                
                np.save(f'./data_fi3d/emb/emb_gen_{key}_{value}.npy', all_motion_gen)
                np.save(f'./data_fi3d/joint/joint_gen_{key}_{value}.npy', all_gen.detach().cpu().numpy())
            
        print("All Done.")
        
        return None
        
    def augment(self, batch_data:torch.Tensor, method, value=1e-3):
        '''
        batch_data: (128, 196, 263)
        '''
        
        batch_data = batch_data.detach().cpu()
        
        if method.lower() == 'none':
            return batch_data
        elif method.lower() == 'uniform':
            delta = value
            min_ = torch.min(batch_data, dim=1, keepdim=True)[0]
            max_ = torch.max(batch_data, dim=1, keepdim=True)[0]
            noise = torch.rand_like(batch_data) * (max_ - min_) + min_
            return batch_data + noise * delta
        elif method.lower() == 'gaussian':
            delta = value
            mean = torch.mean(batch_data, dim=1, keepdim=True)
            std = torch.std(batch_data, dim=1, keepdim=True)
            noise = torch.randn_like(batch_data) * std + mean
            return batch_data + noise * delta
        elif method.lower() == 'smoothing':
            window_size, strength = value[:]
            total_window_size = 2 * window_size + 1
            batch_data_reshaped = batch_data.transpose(1, 2)  # Shape: (128, 263, 196)
            smoothed_data = F.avg_pool1d(batch_data_reshaped, total_window_size, stride=1, padding=window_size, count_include_pad=True) # along the last axis
            smoothed_data = smoothed_data.transpose(1, 2)  # Shape: (128, 196, 263)
            batch_data = (1 - strength) * batch_data + strength * smoothed_data
            return batch_data
        elif method.lower() == 'skipping':
            # randomly delete a frame(along axis 1) for skip_time times, by repeating the last frame
            skip_time = value
            for i in range(skip_time):
                idx = torch.randint(1, batch_data.shape[1], (1,))
                batch_data = torch.cat([batch_data[:, :idx], batch_data[:, idx-1], batch_data[:, idx+1:]], dim=1)
            return batch_data
        elif method.lower() == 'repeating':
            if value == 'half':  # 1/2 fps
                for i in range(batch_data.shape[0]):
                    for j in range((batch_data.shape[1]) // 2):
                        batch_data[i, 2*j+1] = batch_data[i, 2*j]   # [1, 2, 3, 4, 5, 6] -> [1, 1, 3, 3, 5, 5]
            elif value == 'quater':    # 1/4 fps
                for i in range(batch_data.shape[0]):
                    for j in range((batch_data.shape[1]) // 4):
                        batch_data[i, 4*j+1] = batch_data[i, 4*j]
                        batch_data[i, 4*j+2] = batch_data[i, 4*j]
                        batch_data[i, 4*j+3] = batch_data[i, 4*j]
            return batch_data            
        elif method.lower() == 'downframerate':
            # skip every other frame(along axis 1), will change frame rate and sequence length
            batch_data = batch_data[:, ::2]
            return batch_data
        elif method.lower() == 'upframerate':
            # interpolate batch_data along axis 1, will change frame rate and sequence length
            batch_data = batch_data.transpose(1, 2)
            batch_data = torch.nn.functional.interpolate(batch_data, scale_factor=2, mode='linear', align_corners=True)
            return batch_data
        elif method.lower() == 'shuffling':
            # switch two frames(along axis 1) randomly for shuffle_time times
            shuffle_time = value
            for i in range(shuffle_time):
                idx1 = np.random.randint(batch_data.shape[1])
                idx2 = np.random.randint(batch_data.shape[1])
                batch_data[:, idx1], batch_data[:, idx2] = batch_data[:, idx2], batch_data[:, idx1]
            return batch_data
        else:
            raise KeyError('Augmentation method not recognized!!')