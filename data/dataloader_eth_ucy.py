import os
import pickle
import glob
import numpy as np
import math
from einops import rearrange
import torch
import matplotlib.pyplot as plt
from utils.normalization import normalize_min_max


def seq_collate_eth(batch):
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


def rotate_traj(past_rel, future_rel, past_abs, agents=2, rotate_time_frame=0, subset='eth'):
    past_rel = rearrange(past_rel, 'b a p d -> (b a) p d')
    past_abs = rearrange(past_abs, 'b a p d -> (b a) p d')
    future_rel = rearrange(future_rel, 'b a f d -> (b a) f d')
    past_diff = past_rel[:, rotate_time_frame]
    # past_diff = past[:, rotate_time_frame] - past[:, rotate_time_frame-1]
    past_theta = torch.atan(torch.div(past_diff[:, 1], past_diff[:, 0]+1e-5))
    past_theta = torch.where((past_diff[:, 0]<0), past_theta+math.pi, past_theta)
    
    rotate_matrix = torch.zeros((past_theta.size(0), 2, 2)).to(past_theta.device)
    rotate_matrix[:, 0, 0] = torch.cos(past_theta)
    rotate_matrix[:, 0, 1] = torch.sin(past_theta)
    rotate_matrix[:, 1, 0] = - torch.sin(past_theta)
    rotate_matrix[:, 1, 1] = torch.cos(past_theta)
    ### store the inverse of this rotate_matrix
    # inverse_rotate_matrix = rotate_matrix.transpose(1, 2)
    # np.save(f'inverse_rotate_matrix_{subset}.npy', inverse_rotate_matrix.detach().cpu().numpy())
    # exit()

    past_after = torch.matmul(rotate_matrix, past_rel.transpose(1, 2)).transpose(1, 2)
    future_after = torch.matmul(rotate_matrix, future_rel.transpose(1, 2)).transpose(1, 2)
    past_abs_after = torch.matmul(rotate_matrix, past_abs.transpose(1, 2)).transpose(1, 2)
    past_after = rearrange(past_after, '(b a) p d -> b a p d', a=agents)
    future_after = rearrange(future_after, '(b a) f d -> b a f d', a=agents)
    past_abs_after = rearrange(past_abs_after, '(b a) p d -> b a p d', a=agents)

    

    return past_after, future_after, past_abs_after


class ETHDataset(object):
    def __init__(self, cfg, training=True, data_dir = None, subset = None, rotate_time_frame=0, imle=False, type='original'):
        ### LED version of preprocessed data
        if type == 'LED':
            data_file_path = os.path.join(data_dir, type, '{:s}_data_{:s}.npy'.format(subset, 'train' if training else 'test'))
            num_file_path = os.path.join(data_dir, type, '{:s}_num_{:s}.npy'.format(subset, 'train' if training else 'test'))

            all_data = np.load(data_file_path)
            all_num = np.load(num_file_path)
            self.all_data = torch.Tensor(all_data)
            self.all_num = torch.Tensor(all_num) 
        elif type == 'original':
            ### Original version of preprocessed data
            data_file_path = os.path.join(data_dir, type, subset,'{:s}_{:s}.pkl'.format(subset, 'train' if training else 'test'))
            all_data = pickle.load(open(data_file_path, 'rb'))
            all_data = all_data['traj']
            self.all_data = torch.Tensor(all_data)   #[A, T, 2]
            self.all_data = self.all_data[:,None,:,:]  #[A, 1, T, 2]
        else:
            raise ValueError('Invalid type')


        self.cfg = cfg
        self.rotate_time_frame = rotate_time_frame
        self.imle = imle

        
        ### set the agent_num in the cfg
        cfg.agents = self.all_data.shape[1]
        cfg.MODEL.CONTEXT_ENCODER.AGENTS = cfg.agents
        
        ### compute past and future trajectories
        past_traj_abs = self.all_data[:,:,:cfg.past_frames]
        initial_pos = past_traj_abs[:, :, -1:]
        past_traj_rel = (past_traj_abs - initial_pos).contiguous()
        fut_traj = (self.all_data[:,:,cfg.past_frames:] - initial_pos).contiguous()
        if cfg.rotate:
            past_traj_rel, fut_traj, past_traj_abs = rotate_traj(past_traj_rel, fut_traj, past_traj_abs, cfg.agents, rotate_time_frame, subset)
        past_traj_vel = torch.cat((past_traj_rel[:, :, 1:] - past_traj_rel[:, :, :-1], torch.zeros_like(past_traj_rel[:,:, -1:])), dim=2)
        past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)
        self.fut_traj_vel = torch.cat((fut_traj[:, :, 1:] - fut_traj[:,:, :-1], torch.zeros_like(fut_traj[:, :, -1:])), dim=2)

        self.rotate_aug = cfg.rotate_aug and training

        if training:
            cfg.fut_traj_max = fut_traj.max()
            cfg.fut_traj_min = fut_traj.min()
            cfg.past_traj_max = past_traj.max()
            cfg.past_traj_min = past_traj.min()
        
        ### record the original to avoid numerical errors
        self.past_traj_original_scale = past_traj
        self.fut_traj_original_scale = fut_traj
   
        ### min-max linear normalization
        if cfg.data_norm == 'min_max':
            self.past_traj = normalize_min_max(past_traj, cfg.past_traj_min, cfg.past_traj_max, -1, 1).contiguous()
            self.fut_traj = normalize_min_max(fut_traj, cfg.fut_traj_min, cfg.fut_traj_max, -1, 1).contiguous()
        elif cfg.data_norm == 'original':
            self.past_traj = past_traj
            self.fut_traj = fut_traj


        """load distillation target"""
        if imle:
            os.makedirs(os.path.join(data_dir, f'imle/{subset}'), exist_ok=True)
            pkl_ls = sorted(glob.glob(os.path.join(data_dir, f'imle/{subset}/*train*.pkl')))

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
                            
                    # past_tarj_original_scale_ = torch.from_numpy(data['past_traj_original_scale'])
                    # assert torch.sum(torch.abs(past_tarj_original_scale_[:10] - self.past_traj_original_scale[:10])) < 1e-5, 'IMLE data is not consistent'

                    pass

            # concat the data
            for key in keys_ls:
                imle_data_dict[key] = torch.from_numpy(np.concatenate(imle_data_dict[key], axis=0))[:len(self.past_traj)]

            self.imle_data_dict = imle_data_dict

    def __len__(self):
        return self.all_data.shape[0]

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

            if self.rotate_aug:
                A = past_traj_norm_scale.size(0)
                rot_angle = torch.rand(A) * 2 * math.pi

                past_traj_abs_o, past_traj_rel_o, past_traj_vel_o = past_traj_original_scale.chunk(3, dim=-1)   # [A, P, 2] * 3

                rotate_matrix = torch.zeros((rot_angle.size(0), 2, 2)).to(past_traj_norm_scale.device)          # [A, 2, 2]
                rotate_matrix[:, 0, 0] = torch.cos(rot_angle)
                rotate_matrix[:, 0, 1] = torch.sin(rot_angle)
                rotate_matrix[:, 1, 0] = - torch.sin(rot_angle)
                rotate_matrix[:, 1, 1] = torch.cos(rot_angle)

                past_traj_abs_o_rot = torch.matmul(rotate_matrix, past_traj_abs_o.transpose(1, 2)).transpose(1, 2)
                past_traj_rel_o_rot = torch.matmul(rotate_matrix, past_traj_rel_o.transpose(1, 2)).transpose(1, 2)
                fut_traj_o_rot = torch.matmul(rotate_matrix, fut_traj_original_scale.transpose(1, 2)).transpose(1, 2)

                past_traj_vel_o_rot = torch.cat((past_traj_rel_o_rot[:, :, 1:] - past_traj_rel_o_rot[:, :, :-1], torch.zeros_like(past_traj_rel_o_rot[:,:, -1:])), dim=2)
                past_traj_rot = torch.cat((past_traj_abs_o_rot, past_traj_rel_o_rot, past_traj_vel_o_rot), dim=-1)
                
                fut_traj_vel_o = torch.cat((fut_traj_o_rot[:, :, 1:] - fut_traj_o_rot[:,:, :-1], torch.zeros_like(fut_traj_o_rot[:, :, -1:])), dim=2)

                
                past_traj_norm_scale = normalize_min_max(past_traj_rot, self.cfg.past_traj_min, self.cfg.past_traj_max, -1, 1).contiguous()
                fut_traj_norm_scale = normalize_min_max(fut_traj_o_rot, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1).contiguous()
                past_traj_original_scale = past_traj_rot
                fut_traj_original_scale = fut_traj_o_rot
                fut_traj_vel = fut_traj_vel_o

            out = [
                torch.Tensor([item]).to(torch.int32),
                past_traj_norm_scale,
                fut_traj_norm_scale,
                past_traj_original_scale,
                fut_traj_original_scale, 
                fut_traj_vel
            ]
        return out

class ETHDatasetSocialGAN:
    '''
    For the purpose of saving pickle files for train/val/test data for ETH-UCY dataset
    '''
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, min_ped=1, delim='\t', subset='eth'):
        super(ETHDatasetSocialGAN, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.cur_frame_no = 0

        all_files_train = sorted(os.listdir(self.data_dir + '/train/'))
        all_files_test = sorted(os.listdir(self.data_dir + '/test/'))
        all_files_val = sorted(os.listdir(self.data_dir + '/val/'))
        all_files_train = [os.path.join(self.data_dir + '/train/', path) for path in all_files_train]
        all_files_test = [os.path.join(self.data_dir + '/test/', path) for path in all_files_test]
        all_files_val = [os.path.join(self.data_dir + '/val/', path) for path in all_files_val]
        data_dict = {'train': all_files_train, 'test': all_files_test, 'val': all_files_val}
        
        
        for type, all_files in data_dict.items():

            ### init the frame, seq_list and num_peds_in_seq while storing the train/val/test data
            num_peds_in_seq = []
            seq_list = []
            frame_list = []
            for path in all_files:    
                data = self.read_file(path, delim)
                frames = np.unique(data[:, 0]).tolist()
                frame_data = []
                for frame in frames:
                    frame_data.append(data[frame == data[:, 0], :])
                num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

                for idx in range(0, num_sequences * self.skip + 1, skip):
                    curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                    self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                    curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))

                    num_peds_considered = 0
                    for _, ped_id in enumerate(peds_in_curr_seq):
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                        curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                        pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                        pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                        if pad_end - pad_front != self.seq_len:
                            continue
                        curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                        _idx = num_peds_considered
                        curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                        num_peds_considered += 1
                    if num_peds_considered > min_ped:
                        num_peds_in_seq.append(num_peds_considered)
                        seq_list.append(curr_seq[:num_peds_considered])
                        frame_list.append(frames[idx])

            self.num_seq = len(seq_list)
            self.trajs = np.concatenate(seq_list, axis=0).transpose(0, 2, 1)
            cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
            self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
            self.frame_list = np.array(frame_list, dtype=np.int32)
            traj_data = {
                'traj': self.trajs,
                'frame_list': self.frame_list,
                'num_peds_in_seq': num_peds_in_seq,
                'seq_start_end': self.seq_start_end
            }
            pickle.dump(traj_data, open(self.data_dir + '/{:s}_{:s}.pkl'.format(subset, type), 'wb'))


    def read_file(self, _path, delim):
        delim = delim if delim else self.delim
        data = []
        with open(_path, 'r') as f:
            for line in f:
                line = line.strip().split(delim)
                line = [float(i) for i in line]
                data.append(line)
        return np.asarray(data)

