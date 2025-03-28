import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
from torch.utils.data import Dataset
from utils.normalization import normalize_min_max, unnormalize_min_max, normalize_sqrt, unnormalize_sqrt
import torch
from utils.utils import rotate_trajs_x_direction


def seq_collate_nba(batch):
    (past_traj, fut_traj, past_traj_orig, fut_traj_orig, traj_vel) = zip(*batch)
    pre_motion_3D = torch.stack(past_traj,dim=0)
    fut_motion_3D = torch.stack(fut_traj,dim=0)
    pre_motion_3D_orig = torch.stack(past_traj_orig, dim=0)
    fut_motion_3D_orig = torch.stack(fut_traj_orig, dim=0)
    fut_traj_vel = torch.stack(traj_vel, dim=0)

    batch_size = torch.tensor(pre_motion_3D.shape[0]) ### bt 
    traj_mask = torch.zeros(batch_size * 11, batch_size * 11)
    for i in range(batch_size):
        traj_mask[i*11:(i+1)*11, i*11:(i+1)*11] = 1.
    data = {
        'batch_size': batch_size,
        'past_traj': pre_motion_3D,
        'fut_traj': fut_motion_3D,
        'past_traj_original_scale': pre_motion_3D_orig,
        'fut_traj_original_scale': fut_motion_3D_orig,
        'traj_mask': traj_mask,
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
    traj_mask = torch.zeros(batch_size * 11, batch_size * 11)
    for i in range(batch_size):
        traj_mask[i*11:(i+1)*11, i*11:(i+1)*11] = 1.
    data = {
        'batch_size': batch_size,
        'past_traj': pre_motion_3D,
        'fut_traj': fut_motion_3D,
        'past_traj_original_scale': pre_motion_3D_orig,
        'fut_traj_original_scale': fut_motion_3D_orig,
        'fut_traj_vel': fut_traj_vel,
        'traj_mask': traj_mask,
        'y_t': y_t,
        'y_pred_data': y_pred_data
    }

    return data


class NBADatasetMinMax(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self,
        obs_len=5, 
        pred_len=10, 
        training=True, 
        num_scenes=32500, 
        test_scenes=12500,
        overfit=False, 
        traj_scale_total=94/28,
        imle=False,
        cfg=None,
        data_dir='/data/nba',
        rotate=False,
        data_norm='min_max'
    ):
        """
        Args:
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - imle: Whether we train with IMLE or not (a switch)
        """

        super(NBADatasetMinMax, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.imle = imle 
        self.traj_mean = torch.FloatTensor(cfg.traj_mean).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        if not overfit:
            if training:
                data_root = os.path.join(data_dir, 'original/nba_train.npy')
            else:
                data_root = os.path.join(data_dir, 'original/nba_test.npy')
        else:
            data_root = os.path.join(data_dir, 'original/nba_train.npy')

        self.trajs_raw = np.load(data_root) #(N,15,11,2)
        self.trajs = self.trajs_raw / traj_scale_total
        if training:
            self.trajs = self.trajs[:num_scenes]
        else:
            self.trajs = self.trajs[:test_scenes]

        ### Overfit test 
        if overfit:
            self.trajs = self.trajs[:num_scenes]
        

        self.data_len = len(self.trajs)
        print("Size of the dataset: {} in {} mode".format(self.data_len, 'Training' if training else 'Testing'))

        self.traj_abs = torch.from_numpy(self.trajs).type(torch.float)

        self.traj_abs = self.traj_abs.permute(0,2,1,3)
        self.actor_num = self.traj_abs.shape[1]

        pre_motion_3D = self.traj_abs[:, :, :self.obs_len, :]
        fut_motion_3D = self.traj_abs[:, :, self.obs_len:, :]
        initial_pos = pre_motion_3D[:, :, -1:]

        # augment input: absolute position, relative position, velocity
        fut_traj = (fut_motion_3D - initial_pos).contiguous()
        past_traj_abs = (pre_motion_3D - self.traj_mean).contiguous()
        past_traj_rel = (pre_motion_3D - initial_pos).contiguous() 
        if rotate:
            past_traj_rel, fut_traj, past_traj_abs = rotate_trajs_x_direction(past_traj_rel, fut_traj, past_traj_abs)
        past_traj_vel = torch.cat((past_traj_rel[:, 1:] - past_traj_rel[:, :-1], torch.zeros_like(past_traj_rel[:, -1:])), dim=1) 
        past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)
        self.fut_traj_vel = torch.cat((fut_traj[:, 1:] - fut_traj[:, :-1], torch.zeros_like(fut_traj[:, -1:])), dim=1)
        
        if training:
            cfg.fut_traj_max = fut_traj.max()
            cfg.fut_traj_min = fut_traj.min()
            cfg.past_traj_max = past_traj.max()
            cfg.past_traj_min = past_traj.min()
        
        ### record the original to avoid numerical errors
        self.past_traj_original_scale = past_traj
        self.fut_traj_original_scale = fut_traj
        
        self.data_norm = data_norm
        if data_norm == 'min_max':
            ### min-max linear normalization
            self.past_traj = normalize_min_max(past_traj, cfg.past_traj_min, cfg.past_traj_max, -1, 1).contiguous()
            self.fut_traj = normalize_min_max(fut_traj, cfg.fut_traj_min, cfg.fut_traj_max, -1, 1).contiguous()
        elif data_norm == 'sqrt':
            ### sqrt normalization
            sqrt_a_ = torch.tensor([cfg.sqrt_x_a, cfg.sqrt_y_a], device=past_traj.device)
            sqrt_b_ = torch.tensor([cfg.sqrt_x_b, cfg.sqrt_y_b], device=past_traj.device)

            # no need to normalize the past trajectory
            self.past_traj = past_traj


            self.fut_traj = normalize_sqrt(fut_traj, sqrt_a_, sqrt_b_).contiguous()

            # unnormalized_xy = unnormalize_sqrt(self.fut_traj, sqrt_a_, sqrt_b_)
            # unnorm_error = torch.abs(unnormalized_xy - fut_traj).mean()

        """load distillation target"""
        if imle:
            pkl_ls = sorted(glob.glob(os.path.join(data_dir, 'imle/*train*.pkl')))

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

                if total_scenes_loaded_ >= len(self.trajs):
                    break

                # y_t_original_scale_ = unnormalize_min_max(torch.from_numpy(data['y_t'][:, -1]), cfg.fut_traj_min, cfg.fut_traj_max, -1, 1)
                # y_pred_data_original_scale_ = torch.from_numpy(data['y_pred_data'])
                # assert torch.sum(torch.abs(y_t_original_scale_ - y_pred_data_original_scale_)) < 1e-5, 'IMLE data is not consistent'
                        
                # past_tarj_original_scale_ = torch.from_numpy(data['past_traj_original_scale'])
                # assert torch.sum(torch.abs(past_tarj_original_scale_[:10] - self.past_traj_original_scale[:10])) < 1e-5, 'IMLE data is not consistent'

            # concat the data
            for key in keys_ls:
                imle_data_dict[key] = torch.from_numpy(np.concatenate(imle_data_dict[key], axis=0))[:len(self.trajs)]

            self.imle_data_dict = imle_data_dict


    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        if self.imle:
            out = [
                    self.imle_data_dict['past_traj'][index], 
                    self.imle_data_dict['fut_traj'][index],
                    self.imle_data_dict['past_traj_original_scale'][index],
                    self.imle_data_dict['fut_traj_original_scale'][index],
                    self.imle_data_dict['fut_traj_vel'][index],
                    self.imle_data_dict['y_t'][index],
                    self.imle_data_dict['y_pred_data'][index]
                ]
        else:
            out = [
                    self.past_traj[index], 
                    self.fut_traj[index],
                    self.past_traj_original_scale[index],
                    self.fut_traj_original_scale[index], 
                    self.fut_traj_vel[index]
                ]
        return out
        
