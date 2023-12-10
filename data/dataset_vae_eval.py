import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
# import spacy

# from torch.utils.data._utils.collate import default_collate


class MotionDataset(data.Dataset):
    def __init__(self, dataroot, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num
        # print('init opt.window_size', self.opt.window_size) # 24
        
        self.data = []
        self.lengths = []
        self.n_files = len(os.listdir(dataroot))
        
        print(f'loading data from {dataroot}')
        for name in tqdm(os.listdir(dataroot)):
            try:
                motion = np.load(pjoin(dataroot, name))[0]
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0])
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass
        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                        joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                                   joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        # return self.cumsum[-1]
        return self.n_files

    def __getitem__(self, index):
        motion = self.data[index]
        m_length = self.lengths[index]

        if m_length > 196:
            # randomly select a snippet of length 196 from motion
            start = random.randint(0, m_length - 196)
            motion = motion[start:start+196]
        elif m_length < 196:
            # pad motion with zeros to length 196
            motion = np.pad(motion, ((0, 196 - m_length), (0, 0)), 'constant', constant_values=0)
        
        "Z Normalization"   ## 使用的是从train_comp_v6的dataset中读取到的数据，已经normalized了
        # motion = (motion - self.mean) / self.std
        
        return motion, m_length