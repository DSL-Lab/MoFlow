import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
from torch.utils.data import Dataset
import torch.nn.functional as F
from utils.normalization import normalize_min_max, unnormalize_min_max, normalize_sqrt, unnormalize_sqrt
import torch
from utils.utils import rotate_trajs_x_direction

def seq_collate_rat(batch):
    (past_traj, fut_traj, past_traj_orig, fut_traj_orig, traj_vel, hist_feats, hist_cond_cue, fut_cond_cue) = zip(*batch)
    pre_motion_3D = torch.stack(past_traj,dim=0)
    fut_motion_3D = torch.stack(fut_traj,dim=0)
    pre_motion_3D_orig = torch.stack(past_traj_orig, dim=0)
    fut_motion_3D_orig = torch.stack(fut_traj_orig, dim=0)
    fut_traj_vel = torch.stack(traj_vel, dim=0)
    hist_cond_cue = torch.stack(hist_cond_cue, dim=0)
    fut_cond_cue = torch.stack(fut_cond_cue, dim=0)
    hist_feats = torch.stack(hist_feats, dim=0)

    batch_size, vertical_size = pre_motion_3D.shape[0], pre_motion_3D.shape[1] ### bt
    traj_mask = torch.zeros(batch_size * vertical_size, batch_size * vertical_size)
    for i in range(batch_size):
        traj_mask[i*vertical_size:(i+1)*vertical_size, i*vertical_size:(i+1)*vertical_size] = 1.
    data = {
        'batch_size': torch.tensor(batch_size),
        'past_traj': pre_motion_3D,
        'fut_traj': fut_motion_3D,
        'past_traj_original_scale': pre_motion_3D_orig,
        'fut_traj_original_scale': fut_motion_3D_orig,
        'traj_mask': traj_mask,
        'fut_traj_vel': fut_traj_vel,
        'hist_feats': hist_feats,
        'hist_cond_cue': hist_cond_cue,
        "fut_cond_cue": fut_cond_cue,
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

    batch_size, vertical_size = pre_motion_3D.shape[0], pre_motion_3D.shape[1] ### bt
    traj_mask = torch.zeros(batch_size * vertical_size, batch_size * vertical_size)
    for i in range(batch_size):
        traj_mask[i*vertical_size:(i+1)*vertical_size, i*vertical_size:(i+1)*vertical_size] = 1.
    data = {
        'batch_size': torch.tensor(batch_size),
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


class BabelDatasetMinMax(Dataset):
    """
    读取你的大鼠关键点数据，组织为 [N, T_total, V=8, 2]，再切 [T_h | T_p]
    """
    def __init__(self,
                 obs_len=12, pred_len=18, training=True,
                 num_scenes=None, test_scenes=None,
                 overfit=False, imle=False, cfg=None,
                 data_dir='data/babel', data_file='hist10pred20/rat_train.npy',
                 data_norm='min_max'):
        super().__init__()
        self.obs_len  = obs_len
        self.pred_len = pred_len
        self.seq_len  = obs_len + pred_len
        self.imle     = imle
        self.data_norm = data_norm

        self.dt = 1/18
        self.head_idx = 5
        self.neck_idx = 6

        if not overfit:
            if training:
                data_root = os.path.join(data_dir, 'babel_train.npy')
                cmd_root = os.path.join(data_dir, 'babel_train_cmd.npy')
            else:
                data_root = os.path.join(data_dir, 'babel_val.npy')
                cmd_root = os.path.join(data_dir, 'babel_val_cmd.npy')
        else:
            data_root = os.path.join(data_dir, 'babel_test.npy')
            cmd_root = os.path.join(data_dir, 'babel_test_cmd.npy')

        self.trajs_raw = np.load(data_root) #(N,15,11,2)
        self.cmd_raw = np.load(cmd_root)

        # self.trajs = self.trajs_raw / traj_scale_total
        if training:
            self.trajs = self.trajs_raw[:num_scenes]
            self.cmd = self.cmd_raw[:num_scenes]
        else:
            self.trajs = self.trajs_raw[:test_scenes]
            self.cmd = self.cmd_raw[:test_scenes]

        ### Overfit test
        if overfit:
            self.trajs = self.trajs_raw[:num_scenes]
            self.cmd = self.cmd_raw[:num_scenes]

        # print("cmd shape = {}".format(self.cmd.shape))

        self.data_len = len(self.trajs)
        self.traj_abs = torch.from_numpy(self.trajs).float()  # [N, T, V, 2]

        # 对齐 NBA 的维序： [N, V, T, 2]
        self.traj_abs = self.traj_abs.permute(0, 2, 1, 3).contiguous()
        self.actor_num = self.traj_abs.shape[1]  # =8

        pre = self.traj_abs[:, :, :self.obs_len, :]     # [N, V, T_h, 2]
        fut = self.traj_abs[:, :, self.obs_len:, :]     # [N, V, T_p, 2]
        init = pre[:, :, -1:].clone()                   # [N, V, 1, 2]

        # 常见三路特征（与 NBA 一致）：绝对/相对/速度
        fut_traj = (fut - init).contiguous()
        past_abs = pre.contiguous()
        past_rel = (pre - init).contiguous()
        past_vel = torch.cat([past_rel[:, :, 1:] - past_rel[:, :, :-1],
                              torch.zeros_like(past_rel[:, :, -1:])], dim=2)
        past_traj = torch.cat([past_abs, past_rel, past_vel], dim=-1)  # [N,V,T_h, 2+2+2=6]
        self.fut_traj_vel = torch.cat([fut_traj[:, :, 1:] - fut_traj[:, :, :-1],
                                       torch.zeros_like(fut_traj[:, :, -1:])], dim=2)

        # 记录 min/max（训练阶段初始化给 cfg）
        # print("Training = {}".format(training))
        # print("cfg.stats = {}".format(cfg.stats))
        if training:
            stats = {}
            # 绝对位置
            abs_xyz = past_abs.reshape(-1, 3)  # [N*T*V, 2]
            stats['abs_mean'] = abs_xyz.mean(0)
            stats['abs_std'] = abs_xyz.std(0) + 1e-6
            stats['abs_min'], stats['abs_max'] = self.robust_minmax(abs_xyz)

            # 相对位移
            rel_xyz = past_rel.reshape(-1, 3)
            stats['rel_mean'] = rel_xyz.mean(0)
            stats['rel_std'] = rel_xyz.std(0) + 1e-6
            stats['rel_min'], stats['rel_max'] = self.robust_minmax(rel_xyz)

            # 速度
            vel_xyz = past_vel.reshape(-1, 3)
            stats['vel_mean'] = vel_xyz.mean(0)
            stats['vel_std'] = vel_xyz.std(0) + 1e-6
            stats['vel_min'], stats['vel_max'] = self.robust_minmax(vel_xyz)

            # 未来相对位移
            fut_xyz = fut_traj.reshape(-1, 3)
            stats['fut_mean'] = fut_xyz.mean(0)
            stats['fut_std'] = fut_xyz.std(0) + 1e-6
            stats['fut_min'], stats['fut_max'] = self.robust_minmax(fut_xyz)

            cfg.stats = stats
            cfg.fut_traj_max  = None
            cfg.fut_traj_min  = None
            cfg.past_traj_max = None
            cfg.past_traj_min = None
        self.past_traj_original_scale = past_traj.clone()
        self.fut_traj_original_scale  = fut_traj.clone()

        # 归一化（保持与 NBA 版本一致）
        if self.data_norm == 'min_max':
            abs_n = self.z(past_abs, torch.tensor(cfg.stats['abs_mean']), torch.tensor(cfg.stats['abs_std']))
            rel_n = self.z(past_rel, torch.tensor(cfg.stats['rel_mean']), torch.tensor(cfg.stats['rel_std']))
            vel_n = self.z(past_vel, torch.tensor(cfg.stats['vel_mean']), torch.tensor(cfg.stats['vel_std']))

            self.past_traj = torch.cat([abs_n, rel_n, vel_n], dim=-1)  # [N,V,T_h, 2+2+2=6]
            self.fut_traj  = self.z(fut_traj, torch.tensor(cfg.stats['fut_mean']), torch.tensor(cfg.stats['fut_std']))
        else:
            # 也可做 sqrt/log 标准化：略
            self.past_traj = past_traj
            self.fut_traj  = fut_traj

        self.data_len = self.traj_abs.shape[0]
        print(f"Babel Dataset: size {self.data_len} | mode={'train' if training else 'test'}")

        # IMLE 蒸馏数据（如使用）
        if imle:
            # 按 NBA 的 pkl 合并逻辑装填 self.imle_data_dict（字段同名）
            # ...（可后续再开）
            pass

    @torch.no_grad()
    def compute_hist_feats(self,
                           past_abs_xy: torch.Tensor,  # [V, T_h, 2]  原尺度位置（建议 cm）
                           dt: float = 1 / 10,  # 真实采样间隔（秒）
                           head_idx: int = 0,
                           neck_idx: int = 1,
                           ) -> torch.Tensor:
        """
        返回 hist_feats: [V, T_h, C_h]
        C_h 顺序:
          [ vel_x, vel_y, speed, acc, yaw, d_yaw, kappa ]
        均保证时间维长度 == T_h
        """
        device = past_abs_xy.device
        V, T, _ = past_abs_xy.shape

        # 1) 每关键点速度/加速度（长度 T）
        vel = _time_derivative(past_abs_xy, dt)  # [V, T, 2]
        acc_vec = _time_derivative(vel, dt)  # [V, T, 2]
        speed = vel.norm(dim=-1, keepdim=True)  # [V, T, 1]
        acc = acc_vec.norm(dim=-1, keepdim=True)  # [V, T, 1]

        # 2) 用 head-neck 估计 yaw / d_yaw / kappa（长度 T）
        head = past_abs_xy[head_idx]  # [T, 2]
        neck = past_abs_xy[neck_idx]  # [T, 2]
        hn = head - neck  # [T, 2]
        yaw = torch.atan2(hn[..., 1], hn[..., 0])  # [T]
        # wrap 到 [-pi, pi]
        yaw = (yaw + torch.pi) % (2 * torch.pi) - torch.pi
        d_yaw = _time_derivative(yaw, dt)  # [T]
        d_yaw = (d_yaw + torch.pi) % (2 * torch.pi) - torch.pi

        # 3) 曲率 kappa（用 head 轨迹，长度 T）
        v = _time_derivative(head, dt)  # [T, 2]
        a = _time_derivative(v, dt)  # [T, 2]
        cross = (v[..., 0] * a[..., 1] - v[..., 1] * a[..., 0]).abs()  # |x'y'' - y'x''|
        denom = (v.pow(2).sum(dim=-1).clamp_min(1e-6)).pow(1.5)  # (x'^2 + y'^2)^(3/2)
        kappa = cross / denom  # [T]

        # 4) 广播到每个关键点（保证 [V, T, 1]）
        yaw_b = yaw.view(1, T, 1).repeat(V, 1, 1)
        dyaw_b = d_yaw.view(1, T, 1).repeat(V, 1, 1)
        kappa_b = kappa.view(1, T, 1).repeat(V, 1, 1)

        # 5) 拼接特征（时间维全是 T，不会短一维）
        hist_feats = torch.cat([vel, speed, acc, yaw_b, dyaw_b, kappa_b], dim=-1)
        # 形状: [V, T, 2 + 1 + 1 + 1 + 1 + 1] = [V, T, 7]
        return hist_feats

    @torch.no_grad()
    def compute_zc(
            self,
            hist_feats: torch.Tensor,  # [V, T_h, C_h]
            cue_feats: torch.Tensor,  # [T_h, C_c]
            stats: dict = None,  # 训练集统计 {'mean': [4], 'std': [4]}，可为 None
    ) -> torch.Tensor:
        """
        Dz = 4: [speed, d_yaw, kappa, signed_strength]
        return: z_c [V, T_h, 4]
        """
        V, T, C = hist_feats.shape
        speed = hist_feats[..., 2]  # [V,T]
        d_yaw = hist_feats[..., 5]  # [V,T]
        kappa = hist_feats[..., 6]  # [V,T]

        signed_strength = torch.zeros(T, device=hist_feats.device)
        if cue_feats is not None and cue_feats.shape[-1] >= 2:
            signed_strength = cue_feats[..., -2]  # 与上面 compute_cue_feats 保持一致
        signed_strength = signed_strength.view(1, T).expand(V, T)

        z = torch.stack([speed, d_yaw, kappa, signed_strength], dim=-1)  # [V,T,4]

        if stats is not None:
            mean = torch.as_tensor(stats['mean'], device=z.device).view(1, 1, 4)
            std = torch.as_tensor(stats['std'], device=z.device).view(1, 1, 4)
            z = (z - mean) / (std + 1e-6)

        return z

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        past = self.past_traj_original_scale[index][..., :2]  # [V,T_h,2] 取绝对坐标那两维
        fut = self.fut_traj_original_scale[index]  # [V,T_p,2]

        # # 2) cue 的原始输入（示例：你应从日志里读到这两条）
        # instr_id = torch.from_numpy(self.cmd[index, :, 0])  # [T_h + T_f] 0/1/2
        # instr_strength = torch.from_numpy(self.cmd[index, :, 1])  # [T_h + T_f] float

        # 3) 计算四个核心特征
        hist_feats = self.compute_hist_feats(past, dt=self.dt, head_idx=self.head_idx, neck_idx=self.neck_idx)  # [V,T_h,C_h]
        cue_feats = torch.from_numpy(self.cmd[index, :, :])
        # z_d_logits = self.compute_zd_logits(hist_feats, cue_feats, s_thr=self.s_thr, r_thr=self.r_thr, tau=self.tau)  # [V,T_h,4]
        # z_c = self.compute_zc(hist_feats, cue_feats, stats=None)  # [V,T_h,4]

        # 与 NBA 返回顺序对齐
        return [
            self.past_traj[index],                 # [V,T_h,6]
            self.fut_traj[index],                  # [V,T_p,2]
            self.past_traj_original_scale[index],  # [V,T_h,6]
            self.fut_traj_original_scale[index],   # [V,T_p,2]
            self.fut_traj_vel[index],              # [V,T_p,2]

            hist_feats,
            cue_feats[:30],  # 历史刺激序列
            cue_feats[30:],  # 未来刺激序列
        ]

    def robust_minmax(self, x, q=(1, 99)):
        lo = np.percentile(x, q[0], axis=0)
        hi = np.percentile(x, q[1], axis=0)
        return lo, hi

    def z(self, x, mean, std):
        # print("Normalize shape = {}".format(x.shape))
        return (x - mean) / std

    def iz(self, zx, mean, std):
        return zx * std + mean

# ---------- 公用小工具 ----------
def _central_diff(x, dt, pad_mode='replicate'):
    """
    x: [..., T, C] ; central difference in time
    return: same shape
    """
    xp = F.pad(x[..., 2:, :], (0,0,1,0), mode=pad_mode)   # t+1
    xm = F.pad(x[..., :-2, :], (0,0,0,1), mode=pad_mode)  # t-1
    return (xp - xm) / (2 * dt)

def _time_derivative(x: torch.Tensor, dt: float, time_dim: int = -2) -> torch.Tensor:
    """
    长度保持不变的一阶时间导数（forward/central/backward 差分混合）

    参数
    ----
    x        : 张量，形状可为 (T,), (T,C), (...,T), (...,T,C)
               约定时间维在 time_dim（当 x 维度>=3 时），否则自动推断：
               - 1D: (T,) 视作 (T,1)
               - 2D: (T,C) 视作 time_dim=0
    dt       : 采样间隔（秒）
    time_dim : x 的时间维（当 x.ndim >= 3 时生效），默认 -2（常见的 (..., T, C)）

    返回
    ----
    与 x 同形状、同 dtype 的导数张量，时间长度与 x 完全一致
    """
    orig = x
    dtype = x.dtype
    squeeze_mode = None

    # 统一到形状 (..., T, C)，便于差分
    if x.ndim == 1:
        # (T,) -> (1, T, 1)
        x = x.view(1, x.shape[0], 1)
        squeeze_mode = "T"
    elif x.ndim == 2:
        # (T, C) -> (1, T, C) （时间维=0）
        x = x.unsqueeze(0)
        squeeze_mode = "TC"
    else:
        # 把时间维移到 -2，通道维留在 -1
        x = x.movedim(time_dim, -2)

    T = x.size(-2)
    dx = torch.empty_like(x, dtype=dtype)

    if T == 1:
        dx.zero_()
    elif T == 2:
        # 两帧：用前/后向差分；为了长度一致，让两端相同
        d = (x[..., 1, :] - x[..., 0, :]) / dt
        dx[..., 0, :] = d
        dx[..., 1, :] = d
    else:
        # 前向差分（首帧）
        dx[..., 0, :]  = (x[..., 1, :]   - x[..., 0, :])   / dt
        # 中心差分（中间帧）
        dx[..., 1:-1, :] = (x[..., 2:, :] - x[..., :-2, :]) / (2.0 * dt)
        # 后向差分（末帧）
        dx[..., -1, :] = (x[..., -1, :]  - x[..., -2, :])  / dt

    # 还原到输入的轴顺序/形状
    if orig.ndim >= 3:
        dx = dx.movedim(-2, time_dim)  # 把时间维移回去
    if squeeze_mode == "T":
        dx = dx.view(-1)               # (T,)
    elif squeeze_mode == "TC":
        dx = dx.squeeze(0)             # (T, C)

    return dx

def _safe_div(a, b, eps=1e-6):
    return a / (b.abs() + eps)

def _wrap_pi(a):
    # wrap angle to [-pi, pi]
    return (a + torch.pi) % (2 * torch.pi) - torch.pi
