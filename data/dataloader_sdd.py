import os
from glob import glob

from einops import rearrange
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle

from utils.normalization import normalize_min_max
from torch.nn.utils.rnn import pad_sequence


def rotate_traj(past_rel, future_rel, past_abs, rotate_time_frame=0):
    """
    @params past_rel: [N, A, P, 2]
    @params future_rel: [N, A, F, 2]
    @params past_abs: [N, A, P, 2]
    @params rotate_time_frame: int
    """

    A = past_rel.size(1)
    past_rel = rearrange(past_rel, 'b a p d -> (b a) p d')
    past_abs = rearrange(past_abs, 'b a p d -> (b a) p d')
    future_rel = rearrange(future_rel, 'b a f d -> (b a) f d')

    past_diff = past_rel[:, rotate_time_frame]
    # past_diff = past[:, rotate_time_frame] - past[:, rotate_time_frame-1]

    past_theta = torch.atan(torch.div(past_diff[:, 1], past_diff[:, 0] + 1e-5))
    past_theta = torch.where((past_diff[:, 0] < 0), past_theta + math.pi, past_theta)
    
    rotate_matrix = torch.zeros((past_theta.size(0), 2, 2)).to(past_theta.device)
    rotate_matrix[:, 0, 0] = torch.cos(past_theta)
    rotate_matrix[:, 0, 1] = torch.sin(past_theta)
    rotate_matrix[:, 1, 0] = -torch.sin(past_theta)
    rotate_matrix[:, 1, 1] = torch.cos(past_theta)

    past_after = torch.matmul(rotate_matrix, past_rel.transpose(1, 2)).transpose(1, 2)          # [N, P, 2]
    future_after = torch.matmul(rotate_matrix, future_rel.transpose(1, 2)).transpose(1, 2)      # [N, F, 2]
    past_abs_after = torch.matmul(rotate_matrix, past_abs.transpose(1, 2)).transpose(1, 2)            # [N, P, 2]

    past_after = rearrange(past_after, '(b a) p d -> b a p d', a=A)
    future_after = rearrange(future_after, '(b a) f d -> b a f d', a=A)
    past_abs_after = rearrange(past_abs_after, '(b a) p d -> b a p d', a=A)


    return past_after, future_after, past_abs_after


def seq_collate_sdd(batch):
    (index, past_traj, fut_traj, past_traj_orig, fut_traj_orig, traj_vel) = zip(*batch)
    indexes = torch.stack(index, dim=0)
    pre_motion_3D = torch.stack(past_traj,dim=0)
    fut_motion_3D = torch.stack(fut_traj,dim=0)
    pre_motion_3D_orig = torch.stack(past_traj_orig, dim=0)
    fut_motion_3D_orig = torch.stack(fut_traj_orig, dim=0)
    fut_traj_vel = torch.stack(traj_vel, dim=0)

    batch_size = torch.tensor(pre_motion_3D.shape[0]) ### bt 
    data = {
        'indexes': indexes,
        'batch_size': batch_size,
        'past_traj': pre_motion_3D,
        'fut_traj': fut_motion_3D,
        'past_traj_original_scale': pre_motion_3D_orig,
        'fut_traj_original_scale': fut_motion_3D_orig,
        'fut_traj_vel': fut_traj_vel,  
    }
    return data 


def seq_collate_imle_train(batch):
    (past_traj, fut_traj, past_traj_orig, fut_traj_orig, traj_vel, y_t, y_pred_data) = zip(*batch)

    pre_motion_3D = torch.stack(past_traj,dim=0)
    fut_motion_3D = torch.stack(fut_traj,dim=0)
    pre_motion_3D_orig = torch.stack(past_traj_orig, dim=0)
    fut_motion_3D_orig = torch.stack(fut_traj_orig, dim=0)
    fut_traj_vel = torch.stack(traj_vel, dim=0)
    y_t = torch.stack(y_t, dim=0)
    y_pred_data = torch.stack(y_pred_data,dim=0)

    batch_size = torch.tensor(pre_motion_3D.shape[0]) ### bt 
    data = {
        'batch_size': batch_size,
        'past_traj': pre_motion_3D,
        'fut_traj': fut_motion_3D,
        'past_traj_original_scale': pre_motion_3D_orig,
        'fut_traj_original_scale': fut_motion_3D_orig,
        'fut_traj_vel': fut_traj_vel,
        'y_t': y_t,
        'y_pred_data': y_pred_data
    }

    return data


class SDDDataset(Dataset):
    def __init__(self, cfg, data_dir, 
                 training=True, overfit=False, rotate_time_frame=0, imle=False):
        super(SDDDataset, self).__init__()

        """init"""
        self.cfg = cfg
        dataset_file = os.path.join(data_dir, 'original/sdd_train.pkl') if training else os.path.join(data_dir, 'original/sdd_test.pkl')
        if overfit:
            dataset_file = os.path.join(data_dir, 'original/sdd_train.pkl')

        ### Compare with NSP model
        # dataset_file = os.path.join(data_dir, 'nsp/sdd_nsp_train.pkl') if training else os.path.join(data_dir, 'nsp/sdd_nsp_test.pkl')
        # if overfit:
        #     dataset_file = os.path.join(data_dir, 'nsp/sdd_nsp_train.pkl')
        ### Compare with NSP model

        self.training = training
        self.overfit = overfit

        self.rotate_time_frame = rotate_time_frame
        self.imle = imle

        self.past_frames = cfg.past_frames
        self.future_frames = cfg.future_frames
        self.seq_len = self.past_frames + self.future_frames
        self.max_agents_per_scene = 0
        assert self.seq_len == 20 and self.past_frames == 8, "Sanity check on frame length failed!"


        """load the data in the original scale"""
        all_data = pickle.load(open(dataset_file, 'rb'))

        print("Mode: {:s}, {:d} sequences".format('train' if training else 'test', len(all_data)))

        """process the data"""
        ### set the agent_num in the cfg
        cfg.MODEL.CONTEXT_ENCODER.AGENTS = cfg.agents
        
        ### compute past and future trajectories
        past_traj_abs = torch.from_numpy(np.stack([scene[0] for scene in all_data], axis=0)).unsqueeze(1)    # [N, 1, T, 2]

        ### Compare with NSP model
        # past_traj_abs = torch.from_numpy(all_data[:,:self.past_frames])[...,:2].unsqueeze(1).float()    # [N, 1, T, 2]
        ### Compare with NSP model

        initial_pos = past_traj_abs[:, :, -1:, :]                                                            # [N, 1, 1, 2]
        past_traj_rel = (past_traj_abs - initial_pos).contiguous()                                           # [N, 1, T, 2]

        fut_traj_abs = torch.from_numpy(np.stack([scene[1] for scene in all_data], axis=0)).unsqueeze(1)     # [N, 1, T, 2]

        ### Compare with NSP model
        # fut_traj_abs = torch.from_numpy(all_data[:,self.past_frames:])[...,:2].unsqueeze(1).float()     # [N, 1, T, 2]
        ### Compare with NSP model 

        fut_traj_rel = (fut_traj_abs - initial_pos).contiguous()                                             # [N, 1, T, 2]

        if cfg.rotate:
            past_traj_rel, fut_traj_rel, past_traj_abs = rotate_traj(past_traj_rel, fut_traj_rel, past_traj_abs, rotate_time_frame)

        past_traj_vel = torch.cat((past_traj_rel[:, :, 1:] - past_traj_rel[:, :, :-1], torch.zeros_like(past_traj_rel[:,:, -1:])), dim=2)
        past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)
        self.fut_traj_vel = torch.cat((fut_traj_rel[:, :, 1:] - fut_traj_rel[:,:, :-1], torch.zeros_like(fut_traj_rel[:, :, -1:])), dim=2)

        self.rotate_aug = cfg.rotate_aug and training

        if training:
            cfg.fut_traj_max = fut_traj_rel.max()
            cfg.fut_traj_min = fut_traj_rel.min()
            cfg.past_traj_max = past_traj.max()
            cfg.past_traj_min = past_traj.min()

        ### record the original to avoid numerical errors
        self.past_traj_original_scale = past_traj
        self.fut_traj_original_scale = fut_traj_rel

        ### min-max normalization to make past_traj in [-1, 1]
        self.past_traj = normalize_min_max(past_traj, cfg.past_traj_min, cfg.past_traj_max, -1, 1).contiguous()

        ### min-max normalization to make fut_traj in [-1, 1]
        self.fut_traj = normalize_min_max(fut_traj_rel, cfg.fut_traj_min, cfg.fut_traj_max, -1, 1).contiguous()


        """load distillation target"""
        if imle:
            pkl_ls = sorted(glob(os.path.join(data_dir, f'imle/*train*.pkl')))

            keys_ls = ['past_traj', 'fut_traj', 'past_traj_original_scale', 'fut_traj_original_scale', 'fut_traj_vel', 'y_t', 'y_pred_data']
            imle_data_dict = {}
            total_scenes_loaded_ = 0   
            for i_pkl, cur_pkl in enumerate(pkl_ls):
                data = pickle.load(open(cur_pkl, 'rb'))

                if i_pkl == 0:
                    self.imle_meta_data = data['meta_data']
                
                for key in keys_ls:
                    if key not in imle_data_dict:
                        imle_data_dict[key] = []
                    if key == 'y_t':
                        imle_data_dict[key].append(data[key][:, -1])
                    else:
                        imle_data_dict[key].append(data[key])

                total_scenes_loaded_ += data['past_traj'].shape[0]

                if total_scenes_loaded_ >= len(self.past_traj):
                    break

                if i_pkl == 0:
                    # y_t_original_scale_ = unnormalize_min_max(torch.from_numpy(data['y_t'][:, -1]), cfg.fut_traj_min, cfg.fut_traj_max, -1, 1)
                    # y_pred_data_original_scale_ = torch.from_numpy(data['y_pred_data'])
                    # assert torch.sum(torch.abs(y_t_original_scale_ - y_pred_data_original_scale_)) < 1e-5, 'IMLE data is not consistent'
                            
                    past_tarj_original_scale_ = torch.from_numpy(data['past_traj_original_scale'])
                    assert torch.sum(torch.abs(past_tarj_original_scale_[:10] - self.past_traj_original_scale[:10])) < 1e-5, 'IMLE data is not consistent'

                    pass

            # concat the data
            for key in keys_ls:
                imle_data_dict[key] = torch.from_numpy(np.concatenate(imle_data_dict[key], axis=0))[:len(self.past_traj)]

            self.imle_data_dict = imle_data_dict
    
    def __len__(self):
        return len(self.past_traj)

    def __getitem__(self, item): 
        if self.imle:
            out = [
                    self.imle_data_dict['past_traj'][item], 
                    self.imle_data_dict['fut_traj'][item],
                    self.imle_data_dict['past_traj_original_scale'][item],
                    self.imle_data_dict['fut_traj_original_scale'][item],
                    self.imle_data_dict['fut_traj_vel'][item],
                    self.imle_data_dict['y_t'][item],
                    self.imle_data_dict['y_pred_data'][item]
                ]
        else:
            ### past traj, future traj, number of pedestrians (presumbly?), index
            past_traj_norm_scale = self.past_traj[item]                             # [A, P, 6]
            fut_traj_norm_scale = self.fut_traj[item]                               # [A, F, 2] 
            past_traj_original_scale = self.past_traj_original_scale[item]          # [A, P, 6]
            fut_traj_original_scale = self.fut_traj_original_scale[item]            # [A, F, 2]
            fut_traj_vel = self.fut_traj_vel[item]                                  # [A, F, 2]   

        
            out = [
                torch.Tensor([item]).to(torch.int32),
                past_traj_norm_scale,
                fut_traj_norm_scale,
                past_traj_original_scale,
                fut_traj_original_scale,
                fut_traj_vel,
            ]
        return out
