from datetime import datetime
import numpy as np
import torch
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from motion_loaders.model_motion_loaders import get_motion_loader
from utils.get_opt import get_opt
from utils.metrics import *
from networks.evaluator_wrapper import EvaluatorModelWrapper
from collections import OrderedDict
from utils.plot_script import *
from scripts.motion_process import *
from utils import paramUtil
from utils.utils import *

from os.path import join as pjoin

def plot_t2m(data, save_dir, captions):
    data = gt_dataset.inv_transform(data)
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), wrapper_opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%(i))
        plot_3d_motion(save_path, paramUtil.t2m_kinematic_chain, joint, title=caption, fps=20)

torch.multiprocessing.set_sharing_strategy('file_system')

def evaluate_matching_score(motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    activation_move_dict = OrderedDict({})
    # print(motion_loaders.keys())
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        all_movement_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                text_embeddings, motion_embeddings, movement_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                print('motion_embeddings.shape', motion_embeddings.shape)       # torch.Size([32, 512])
                print('movement_embeddings.shape', movement_embeddings.shape)   # ([32, 49, 512])
                
                all_motion_embeddings.append(motion_embeddings.cpu().numpy())
                all_movement_embeddings.append(movement_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            all_movement_embeddings = np.concatenate(all_movement_embeddings, axis=0)
            
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings
            activation_move_dict[motion_loader_name] = all_movement_embeddings

        raise Exception('All motions have been saved to ./eval_data')

    return match_score_dict, R_precision_dict, activation_dict, all_motion_embeddings, all_movement_embeddings

def evaluate_fid(groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    gt_movement_embeddings = []
    print('========== Evaluating FID ==========')
    ### obtain ground truth motion embeddings from reading dataset->embedding network
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, sent_lens, motions, m_lens, _ = batch  # batch size is 1
            '''
            motion embeddings is the output of the 2-layer 1D CNN
            get_motion_embeddings
            -> self.movement_encoder
            -> build_models
            -> movement_enc
            -> checkpoint['movement_encoder']
            '''
            motion_embeddings, movement_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,    # torchsize([32, m_length, 22, 3]), 32 motions due to the default batch size
                m_lens=m_lens
            )
            # print('1', motion_embeddings.shape)
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
            gt_movement_embeddings.append(movement_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)     # torchsize([m_length*32, 512])
    gt_movement_embeddings = np.concatenate(gt_movement_embeddings, axis=0)
    
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)       # mu = np.mean(activations, axis=0); cov = np.cov(activations, rowvar=False) 

    ### obtain fake motion embeddings from activation_dict
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    
    return eval_dict


def extract_gt_motion_embeddings(groundtruth_loader):
    """
    Modified from evaluate_fid
    """
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    gt_movement_embeddings = []
    
    print('========== Evaluating FID ==========')
    print('groundtruth_loader size:', len(groundtruth_loader))
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, sent_lens, motions, m_lens, _ = batch  # batch size is 1
            print('motions.shape', motions.shape) # torch.Size([32, 49, 263])
            motion_embeddings, movement_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,    # torchsize([32, m_length, 22, 3]), 32 motions due to the default batch size
                m_lens=m_lens
            )
            # print('2', motion_embeddings.shape) # torch.Size([32, 512])
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
            gt_movement_embeddings.append(movement_embeddings.cpu().numpy())
            
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)     # [m_length*32, 512]
    gt_movement_embeddings = np.concatenate(gt_movement_embeddings, axis=0)

    return gt_motion_embeddings, gt_movement_embeddings


def evaluate_diversity(activation_dict, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(mm_motion_loaders, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(log_file):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({})})
        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():    # <<< fishy
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader
            
            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict, fake_emb, fake_mov_emb = evaluate_matching_score(motion_loaders, f)  #<<<


if __name__ == '__main__':
    dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD01/opt.txt'

    eval_motion_loaders = {
        ################
        ## HumanML3D Dataset##
        ################
        'Comp_v6_KLD01': lambda: get_motion_loader(
            './checkpoints/t2m/Comp_v6_KLD01/opt.txt',
            batch_size, gt_dataset, mm_num_samples, mm_num_repeats, device
        )
    }

    device_id = 0
    device = torch.device('cuda:%d'%device_id if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device_id)

    mm_num_samples = 10    # <<< 100

    mm_num_repeats = 3     # <<< 30
    mm_num_times = 2       # <<< 10

    diversity_times = 300
    replication_times = 1   # <<< 20
    batch_size = 32


    print('========== Loading Dataset ==========')
    gt_loader, gt_dataset = get_dataset_motion_loader(dataset_opt_path, 32, torch.device(0))
    
    print('========== Loading Model Wrapper ==========')
    wrapper_opt = get_opt(dataset_opt_path, device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    log_file = './t2m_evaluation.log'
    print('========== Start Evaluation ==========')
    evaluation(log_file)