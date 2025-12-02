import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn

from collections import namedtuple

from einops import rearrange, reduce, repeat

from tqdm.auto import tqdm
from utils.normalization import unnormalize_min_max, unnormalize_sqrt, unnormalize_mean_std
from utils.utils import apply_mask
from utils.utils import LossBuffer

ModelPrediction = namedtuple('ModelPrediction', ['pred_vel', 'pred_data', 'pred_score'])


# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def pad_t_like_x(t, x):
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class FlowMatcher(nn.Module):
    def __init__(
        self,
        cfg,
        model,
        logger
    ):
        super().__init__()

        # init
        self.cfg = cfg
        self.model = model
        self.logger = logger
        self.handles = []

        self.num_agents = cfg.agents
        self.out_dim = cfg.MODEL.MODEL_OUT_DIM

        self.objective = cfg.objective
        self.sampling_steps = cfg.sampling_steps
        self.solver = cfg.get('solver', 'euler')

        assert cfg.objective in {'pred_vel', 'pred_data'}, 'objective must be either pred_vel or pred_data'
        assert self.cfg.get('LOSS_VELOCITY', False) == False, 'Velocity loss is not supported yet.'

        # special normalization params
        if self.cfg.get('data_norm', None) == 'sqrt':
            self.sqrt_a_ = torch.tensor([self.cfg.sqrt_x_a, self.cfg.sqrt_y_a], device=self.device)
            self.sqrt_b_ = torch.tensor([self.cfg.sqrt_x_b, self.cfg.sqrt_y_b], device=self.device)

        # set up the loss buffer
        self.loss_buffer = LossBuffer(t_min=0, t_max=1.0, num_time_steps=100)
        # register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))


    @property
    def device(self):
        return self.cfg.device
    
    def get_precond_coef(self, t):
        """
        Get preconditioned wrapper coefficients.
        D_theta = alpha_t * x_t + beta_t * F_theta
        @param t: [B]
        """
        coef_1 = t.pow(2) * self.cfg.sigma_data ** 2 + (1-t).pow(2)
        alpha_t = t * self.cfg.sigma_data ** 2 / coef_1
        beta_t = (1 - t) * self.cfg.sigma_data / coef_1.sqrt()

        return alpha_t, beta_t
    
    def get_input_scaling(self, t):
        """
        Get the input scaling factor.
        """
        var_x_t = self.cfg.sigma_data ** 2 * t.pow(2) + (1 - t).pow(2)
        return 1.0 / var_x_t.sqrt().clip(min=1e-4, max=1e4)

    def fm_wrapper_func(self, x_t, t, model_out):
        """
        Build wrapper for network regression output. We don't modify the classification logits.
        We aim to let the wrapper to match the data prediction (x_1 in the flow model).
        @param x_t: 		[B, K, A, F * D]
        @param t: 			[B]
        @param model_out: 	[B, K, A, F * D]
        """
        if self.cfg.fm_wrapper == 'direct':
            return model_out
        elif self.cfg.fm_wrapper == 'velocity':
            t = pad_t_like_x(t, x_t)
            return x_t + (1 - t) * model_out
        elif self.cfg.fm_wrapper == 'precond':
            t = pad_t_like_x(t, x_t)
            alpha_t, beta_t = self.get_precond_coef(t)
            return alpha_t * x_t + beta_t * model_out


    def predict_vel_from_data(self, x1, xt, t):
        """
        Predict the velocity field from the predicted data.
        """
        t = pad_t_like_x(t, x1)
        v = (x1 - xt) / (1 - t)
        return v

    def predict_data_from_vel(self, v, xt, t):
        """
        Predict the data from the predicted velocity field.
        """
        t = pad_t_like_x(t, xt)
        x1 = xt + v * (1 - t)
        return x1

    def fwd_sample_t(self, x0, x1, t):
        """
        Sample the latent space at time t.
        """
        t = pad_t_like_x(t, x0)
        xt = t * x1 + (1 - t) * x0      # simple linear interpolation
        ut = x1 - x0                    # xt derivative w.r.t. t
        return xt, ut

    def get_reweighting(self, t, wrapper=None):
        wrapper = default(wrapper, self.cfg.fm_wrapper)
        if wrapper == 'direct':
            l_weight = torch.ones_like(t)
        elif wrapper == 'velocity':
            l_weight = 1.0 / (1 - t) ** 2
        elif wrapper == 'precond':
            alpha_t, beta_t = self.get_precond_coef(t)
            l_weight = 1.0 / beta_t ** 2
        if self.cfg.fm_rew_sqrt:
            l_weight = l_weight.sqrt()
        l_weight = l_weight.clamp(min=1e-4, max=1e4)
        return l_weight
    
    def get_loss_input(self, y_start_k):
        """
        Prepare the input for the flow matching model training.
        """

        # random time steps to inject noise
        bs = y_start_k.shape[0] # batch size
        if self.cfg.t_schedule == 'uniform':
            t = torch.rand((bs, ), device=self.device)
        elif self.cfg.t_schedule == 'logit_normal':
            # note: this is logit-normal (not log-normal)
            mean_ = self.cfg.logit_norm_mean
            std_ = self.cfg.logit_norm_std
            t_normal_ = torch.randn((bs, ), device=self.device) * std_ + mean_
            t = torch.sigmoid(t_normal_)
        else:
            if '==' in self.cfg.t_schedule:
                # constant_t
                t = float(self.cfg.t_schedule.split('==')[1]) * torch.ones((bs, ), device=self.device)
            else:
                # custom two-stage uniform distribution
                # e.g., 't0.5_p0.3' means with 30% probability, sample from [0, 0.5] uniformly, and with 70% probability, sample from [0.5, 1] uniformly
                cutoff_t = float(self.cfg.t_schedule.split('_')[0][1:])
                prob_1 = float(self.cfg.t_schedule.split('_')[1][1:])

                t_1 = torch.rand((bs, ), device=self.device) * cutoff_t
                t_2 = cutoff_t + torch.rand((bs, ), device=self.device) * (1 - cutoff_t)
                rand_num = torch.rand((bs, ), device=self.device)

                t = t_1 * (rand_num < prob_1) + t_2 * (rand_num >= prob_1)

        assert t.min() >= 0 and t.max() <= 1

        # noise sample
        if self.cfg.tied_noise:
            noise = torch.randn_like(y_start_k[:, 0:1])                                  # [B, 1, T, D]
            noise = noise.expand(-1, self.cfg.denoising_head_preds, -1, -1)              # [B, K, T, D]
        else:
            noise = torch.randn_like(y_start_k)                                          # [B, K, T, D]

        # sample the latent space at time t
        # 构造监督,给出
        x_t, u_t = self.fwd_sample_t(x0=noise, x1=y_start_k, t=t)                        # [B, K, T, D] * 2

        if self.objective == 'pred_data':
            target = y_start_k
        elif self.objective == 'pred_vel':
            target = u_t
        else:
            raise ValueError(f'unknown objective {self.objective}')

        l_weight = self.get_reweighting(t)

        return t, x_t, u_t, target, l_weight

    def model_predictions(self, y_t, x, t, flag_print):
        if self.cfg.fm_in_scaling:
            y_t_in = y_t * pad_t_like_x(self.get_input_scaling(t), y_t)
        else:
            y_t_in = y_t

        # model_out, pred_score = self.model(y_t_in, t, x_data = x)
        model_out = self.model(y_t_in, t, x_data = x)
        pred_score = torch.tensor(0, device=self.device)
        y_data_at_t = self.fm_wrapper_func(y_t, t, model_out)            # [B, K, A, F * D]

        if self.objective == 'pred_vel':
            raise NotImplementedError

        elif self.objective == 'pred_data':
            gt_y_data = rearrange(x['fut_traj'], 'b a f d -> b 1 a (f d)')

            this_t = round(t.unique().item(), 4)

            if flag_print:
                y_data_ = rearrange(y_data_at_t, 'b k a (f d) -> (b a) k f d', f=self.cfg.future_frames)
                gt_y_data = rearrange(gt_y_data, 'b k a (f d) -> (b a) k f d', f=self.cfg.future_frames)

                if self.cfg.get('data_norm', None) == 'min_max':
                    # y_data_metric = unnormalize_min_max(y_data_, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1)
                    # gt_y_data_metric = unnormalize_min_max(gt_y_data, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1)
                    y_data_metric = unnormalize_mean_std(y_data_, self.cfg.stats["fut_mean"], self.cfg.stats["fut_std"], 1)  # [B, K, A, T, D]
                    gt_y_data_metric = unnormalize_mean_std(gt_y_data, self.cfg.stats["fut_mean"], self.cfg.stats["fut_std"], 1)  # [B, K, A, T, D]

                elif self.cfg.get('data_norm', None) == 'sqrt':
                    y_data_metric = unnormalize_sqrt(y_data_, self.sqrt_a_, self.sqrt_b_)
                    gt_y_data_metric = unnormalize_sqrt(gt_y_data, self.sqrt_a_, self.sqrt_b_)
                elif self.cfg.get('data_norm', None) == 'hist10pred20':
                    y_data_metric = y_data_
                    gt_y_data_metric = gt_y_data

                error_metric = (y_data_metric - gt_y_data_metric).abs()  # [B * A, K, F, D]
                batch_min_ade_approx = error_metric.norm(dim=-1, p=2).mean(dim=-1).min(dim=-1).values.mean()
                if this_t == 0.0:
                    self.logger.info("{}".format("-" * 50))
                self.logger.info("Sampling time step: {:.3f}, batch minADE approx: {:.4f}".format(this_t, batch_min_ade_approx))
                # self.logger.info("Sampling time step: {:.3f}".format(this_t))

            pred_vel = self.predict_vel_from_data(y_data_at_t, y_t, t)

        else:
            raise ValueError(f'unknown objective {self.objective}')

        return ModelPrediction(pred_vel, y_data_at_t, pred_score)

    @torch.inference_mode()
    def bwd_sample_t(self, y_t: torch.tensor, t: int, dt: float, x_data: dict, flag_print: bool=False):
        B, K, T, D = y_t.shape

        batched_t = torch.full((B,), t, device=self.device, dtype=torch.float)
        model_preds = self.model_predictions(y_t, x_data, batched_t, flag_print)

        y_next = y_t + model_preds.pred_vel * dt
        return y_next, model_preds.pred_data, model_preds

    @torch.no_grad()
    def sample(self, x_data, num_trajs, return_all_states=False):
        """
        Sample from the model.
        """
        # start with y_T ~ N(0,I), reversed MC to conditionally denoise the traj
        assert num_trajs == self.cfg.denoising_head_preds, 'num_trajs must be equal to denoising_head_preds = {}'.format(self.cfg.denoising_head_preds)
        y_data = None

        batch_size = x_data['batch_size']
        y_t = torch.randn((batch_size, num_trajs, self.num_agents, self.out_dim), device=self.device)
        if self.cfg.tied_noise:
            y_t = y_t[:, :1].expand(-1, self.cfg.denoising_head_preds, -1, -1)

        # sampling loop
        y_data_at_t_ls = []
        t_ls = []
        y_t_ls = []

        if self.solver == 'euler':
            dt = 1.0 / self.sampling_steps
            t_ls = dt * np.arange(self.sampling_steps)
            dt_ls = dt * np.ones(self.sampling_steps)
        elif self.solver == 'lin_poly':
            # linear time growth in the first half with small dt
            # polinomial growth of dt in the second half
            lin_poly_long_step = self.cfg.lin_poly_long_step
            lin_poly_p = self.cfg.lin_poly_p

            n_steps_lin = self.sampling_steps // 2
            n_steps_poly = self.sampling_steps - n_steps_lin

            dt_lin = 1.0 / lin_poly_long_step
            t_lin_ls = dt_lin * np.arange(n_steps_lin)

            def _polynomially_spaced_points(a, b, N, p=2):
                # Generate N points in the interval [a, b] with spacing determined by the power p.
                points = [a + (b - a) * ((i - 1) ** p) / ((N - 1) ** p) for i in range(1, N + 1)]
                return points

            t_poly_start = t_lin_ls[-1] + dt_lin
            t_poly_end = 1.0
            t_poly_ls_ = _polynomially_spaced_points(t_poly_start, t_poly_end, n_steps_poly + 1, p=lin_poly_p)
            dt_poly = np.diff(t_poly_ls_)

            dt_ls = np.concatenate([dt_lin * np.ones(n_steps_lin), dt_poly]).tolist()
            t_ls = np.concatenate([t_lin_ls, t_poly_ls_[:-1]]).tolist()

        else:
            raise NotImplementedError(f"Unknown solver: {self.solver}")

        # define the time steps to print
        num_prints = 10
        if len(t_ls) > num_prints:
            print_times = t_ls[::self.sampling_steps // num_prints]
            if t_ls[-1] not in print_times:
                print_times.append(t_ls[-1])
        else:
            print_times = t_ls

        for idx_step, (cur_t, cur_dt) in enumerate(zip(t_ls, dt_ls)):
            flag_print = cur_t in print_times
            y_t, y_data, model_preds = self.bwd_sample_t(y_t, cur_t, cur_dt, x_data, flag_print)
            y_data_at_t_ls.append(y_data)
            if return_all_states:
                y_t_ls.append(y_t)

        y_data_at_t_ls = torch.stack(y_data_at_t_ls, dim=1)     # [B, S, K, A, F * D]
        t_ls = torch.tensor(t_ls, device=self.device)   # [S]
        if return_all_states:
            y_t_ls = torch.stack(y_t_ls, dim=1)  # [B, S, K, A, F * D]

        return y_t, y_data_at_t_ls, t_ls, y_t_ls, model_preds.pred_score

    def p_losses(self, x_data, log_dict=None):
        """
        Denoising model training.
        训练一次 denoising / flow-matching 步，返回总损失与各子损失。
        x_data 里通常包含：
          - 'fut_traj': [B, A, T, D]  未来轨迹真值（已归一化）；A=agent数，T=future_frames，D=2(x,y)
          - （可能还有历史、掩码等，用于 self.model 的条件）
        """

        # ---------- 初始化与维度 ----------
        B, A = x_data['fut_traj'].shape[:2] # 批大小B、场景内agent数A
        K = self.cfg.denoising_head_preds   # 每个样本的候选/采样条数（多模态分支数）
        T = self.cfg.future_frames          # 未来帧数
        assert self.objective == 'pred_data', 'only pred_data is supported for now'
        # 当前实现只支持直接预测数据（而非预测噪声/速度之外的目标）

        # ---------- 前向扩散/路径采样：构造 y_t、u_t ----------
        # 把未来真值复制K份以便“best-of-K”训练；展平(T,D)->(T*D)便于与模型接口对齐
        fut_traj_normalized = repeat(x_data['fut_traj'],   # [B, A, T, D]
                                     'b a f d -> b k a (f d)', k=K) # 变成 [B, K, A, T*D]
        # print("fut_traj_normalized shape = {}".format(fut_traj_normalized.shape))
        # 从损失输入构造器拿到时间t、中间状态y_t、目标速度/残差u_t、（占位返回）、损失权重l_weight
        # 常见返回：t:[B], y_t:[B,K,A,T*D], u_t:[B,K,A,T*D], l_weight:[B]或[B,A]
        # 真的,这边变成y_t了
        t, y_t, u_t, _, l_weight = self.get_loss_input(y_start_k = fut_traj_normalized)

        # ---------- 模型前向输入尺度调整（可选） ----------
        if self.cfg.fm_in_scaling:
            # 某些FM变体会把输入按t缩放；pad_t_like_x用于把t的标量编码扩展/对齐到y_t的形状
            y_t_in = y_t * pad_t_like_x(self.get_input_scaling(t), y_t)
        else:
            y_t_in = y_t

        # ---------- 随机输入丢弃（可选，类似CFG/抗过拟合） ----------
        if self.training and self.cfg.get('drop_method', None) == 'input':
            # 逻辑型丢弃概率：p_m = sigmoid(k*(t-m))，t越大越容易被置零（或反之，取决于m,k）
            assert self.cfg.get('drop_logi_k', None) is not None and self.cfg.get('drop_logi_m', None) is not None
            m, k = self.cfg.drop_logi_m, self.cfg.drop_logi_k
            p_m = 1 / (1 + torch.exp(-k * (t - m)))   # [B]，随t变化的丢弃概率
            p_m = p_m[:, None, None, None]            # 扩展到 [B,1,1,1]，便于与 y_t_in 广播
            y_t_in = y_t_in.masked_fill(torch.rand_like(p_m) < p_m, 0.)    # 以概率 p_m 将 y_t_in 掩蔽为0，产生“难例/缺失”训练

        # ---------- 模型前向 ----------
        # self.model: 输入（噪声/中间态 y_t_in, 时间 t, 上下文 x_data），输出：
        #   - model_out: [B, K, A, T*D]  速度/残差/去噪方向（取决于 fm_wrapper_func 的定义）
        #   - denoiser_cls: [B, K, A]    每个候选分支的logits，用于选择/分类损失（可选）
        model_out = self.model(y_t_in, t, x_data=x_data)  # [B, K, A, T * D] + [B, K, A]
        # 把网络输出包一层“FM包装器”：根据FM定义把 model_out 映射到“去噪后的 y”（如 ŷ_0 或 ŷ_{t-Δ}）
        # 常见：Rectified Flow时 denoised_y = y_t + Δt * v_theta(x_t,t)；或 Wrapper 把速度场转换为数据空间
        denoised_y = self.fm_wrapper_func(y_t, t, model_out) # [B, K, A, T*D]

        # ---------- 还原形状，进入度量空间（反归一化） ----------
        # 把最后一维 T*D 还原成 [T, D]
        denoised_y = rearrange(denoised_y, 'b k a (f d) -> b k a f d', f=self.cfg.future_frames)
        # 同样处理GT（注意 fut_traj_normalized 此刻是 [B,K,A,T*D]，还原成 [B,K,A,T,2]）
        fut_traj_normalized = fut_traj_normalized.view(B, K, A, T, -1)
        # 根据 data_norm 反归一化到“评估/物理单位”（像素或厘米）
        if self.cfg.get('data_norm', None) == 'min_max':
            # denoised_y_metric = unnormalize_min_max(denoised_y, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1) 		 # [B, K, A, T, D]
            # fut_traj_metric = unnormalize_min_max(fut_traj_normalized, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1)  # [B, K, A, T, D]
            denoised_y_metric = unnormalize_mean_std(denoised_y, self.cfg.stats["fut_mean"], self.cfg.stats["fut_std"], 0) 		 # [B, K, A, T, D]
            fut_traj_metric = unnormalize_mean_std(fut_traj_normalized, self.cfg.stats["fut_mean"], self.cfg.stats["fut_std"], 0)  # [B, K, A, T, D]
        elif self.cfg.get('data_norm', None) == 'sqrt':
            denoised_y_metric = unnormalize_sqrt(denoised_y, self.sqrt_a_, self.sqrt_b_)            # [B, K, A, T, D]
            fut_traj_metric = unnormalize_sqrt(fut_traj_normalized, self.sqrt_a_, self.sqrt_b_)     # [B, K, A, T, D]
        elif self.cfg.get('data_norm', None) == 'hist10pred20':
            denoised_y_metric = denoised_y
            fut_traj_metric = fut_traj_normalized
        else:
            raise ValueError(f"Unknown data normalization method: {self.cfg.get('data_norm', None)}")

        # ---------- 速度正则（未实现分支，占位） ----------
        if self.cfg.get('LOSS_VELOCITY', False):
            raise NotImplementedError
            denoised_y_metric = rearrange(denoised_y_metric, 'b k a (f d) -> b k a f d', f = self.cfg.future_frames, d = 4)
            denoised_y_metric_xy, denoised_y_metric_v = denoised_y_metric[..., :2], denoised_y_metric[..., 2:4]

            gt_traj_vel = x_data['fut_traj_vel'][:, None].expand(-1, K, -1, -1, -1)  # [B, K, A, T, 2]
            loss_reg_vel = F.l1_loss(denoised_y_metric_v, gt_traj_vel, reduction='none').mean()
        else:
            denoised_y_metric_xy = denoised_y_metric
            loss_reg_vel = torch.zeros(1).to(self.device)

        # ---------- 均值绝对误差（或平方误差） ----------
        # 逐agent逐帧L2误差：||pred-gt||_2，形状 [B,K,A,T]
        denoising_error_per_agent = (denoised_y_metric_xy - fut_traj_metric).view(B, K, A, T, -1).norm(dim=-1)  	 # [B, K, A, T]

        # 可选：把L2误差平方，等价于 MSE 的根去掉（按需求）
        if self.cfg.get('LOSS_REG_SQUARED', False):
            denoising_error_per_agent = denoising_error_per_agent ** 2

        # 聚合到“场景级”误差：对 agent 维 A 求平均 ⇒ [B,K,T]
        denoising_error_per_scene = denoising_error_per_agent.mean(dim=-2)  								 	 # [B, K, T]

        if self.cfg.get('LOSS_REG_REDUCTION', 'mean') == 'mean':
            # scene: [B,K]; agent: [B,K,A]
            denoising_error_per_scene = denoising_error_per_scene.mean(dim=-1)
            denoising_error_per_agent = denoising_error_per_agent.mean(dim=-1)
        elif self.cfg.get('LOSS_REG_REDUCTION', 'mean') == 'sum':
            denoising_error_per_scene = denoising_error_per_scene.sum(dim=-1)
            denoising_error_per_agent = denoising_error_per_agent.sum(dim=-1)
        else:
            raise ValueError(f"Unknown reduction method: {self.cfg.get('LOSS_REG_REDUCTION', 'mean')}")

        # ---------- 组件选择（“赢家通吃” best-of-K），同时训练分类头 ----------
        if self.cfg.LOSS_NN_MODE == 'scene':
            # 1) 在场景级上选择：对每个样本取使 scene 误差最小的那个K分支
            selected_components = denoising_error_per_scene.argmin(dim=1)  # [B]
            # 取出对应k*的场景误差，作为回归损失基数
            loss_reg_b = denoising_error_per_scene.gather(1, selected_components[:, None]).squeeze(1)  		# [B]

            # 分类头：预测哪个K是最佳（对A平均后为 [B,K]）
            # cls_logits = denoiser_cls.mean(dim=-1)  # [B, K]
            loss_cls_b = F.cross_entropy(input=cls_logits, target=selected_components, reduction='none')	# [B]
        elif self.cfg.LOSS_NN_MODE == 'agent':
            # agent-level selection
            # 2) 在agent级上选择：每个agent各自找最优k* ⇒ [B,A]
            selected_components = denoising_error_per_agent.argmin(dim=1)  # [B, A]
            # 按k*提取每个agent的误差 ⇒ [B,A]
            loss_reg_b = denoising_error_per_agent.gather(1, selected_components[:, None, :]).squeeze(1)  	# [B, A]
            # 再对A平均成 [B]
            loss_reg_b = loss_reg_b.mean(dim=-1)  # [B]
            # 分类头：把 [B,K,A] 拉平为 [B*A,K]，每个agent各自做分类
            # cls_logits = rearrange(denoiser_cls, 'b k a -> (b a) k')	# [B * A, K]
            # cls_labels = selected_components.view(-1)					# [B * A]
            # loss_cls_b = F.cross_entropy(input=cls_logits, target=cls_labels, reduction='none')	 # [B * A]
            # loss_cls_b = loss_cls_b.view(B, A).mean(dim=-1)  	# [B]
        elif self.cfg.LOSS_NN_MODE == 'both':
            # 3) 同时做场景级与agent级的选择，并线性组合
            selected_components = denoising_error_per_scene.argmin(dim=1)  # [B]
            loss_reg_b_scene = denoising_error_per_scene.gather(1, selected_components[:, None]).squeeze(1)  		# [B] 

            # agent-level selection
            selected_components = denoising_error_per_agent.argmin(dim=1)  # [B, A]
            loss_reg_b = denoising_error_per_agent.gather(1, selected_components[:, None, :]).squeeze(1)  	# [B, A]
            loss_reg_b_agent = loss_reg_b.mean(dim=-1)  # [B]
            # 线性权重 omega：场景级占比
            loss_reg_b = self.cfg.OPTIMIZATION.LOSS_WEIGHTS.get('omega', 1.0)  * loss_reg_b_scene + loss_reg_b_agent

            ## 分类损失占位（如未训练分类头）
            # loss_cls_b = torch.zeros_like(loss_reg_b)

        # ---------- 组装总损失 ----------
        # 回归损失：按时间层级/噪声层级的权重 l_weight（通常来自 t 或路径调度）加权
        # l_weight 需与 [B] 对齐（若是 [B,A] 也应在上面对应求均值后再乘）
        loss_reg = (loss_reg_b * l_weight).mean()  # scalar
        # 分类损失：平均到 batch 标量
        # loss_cls = loss_cls_b.mean()
        loss_cls = np.array([0])
        # 各项权重
        weight_reg = self.cfg.OPTIMIZATION.LOSS_WEIGHTS.get('reg', 1.0)
        weight_cls = self.cfg.OPTIMIZATION.LOSS_WEIGHTS.get('cls', 1.0)
        weight_vel = self.cfg.OPTIMIZATION.LOSS_WEIGHTS.get('vel', 0.2)
        # 最终总损失：回归 + 分类 + 速度正则
        loss = weight_reg * loss_reg.mean() + weight_cls * loss_cls.mean() + weight_vel * loss_reg_vel.mean()

        # ---------- 分噪声级别(loss per level) 记录 ----------
        # 记录不同 t（或噪声等级）的损失曲线，便于诊断FM在各时间的学习情况
        flag_reset = self.loss_buffer.record_loss(t, loss_reg_b.detach(), epoch_id=log_dict['cur_epoch'])
        if flag_reset:
            dict_loss_per_level = self.loss_buffer.get_average_loss()
            log_dict.update({
                'denoiser_loss_per_level': dict_loss_per_level
            })

        return loss, loss_reg.mean(), loss_cls.mean(), loss_reg_vel.mean()

    def forward(self, x, log_dict=None):
        self.cfg.stats["fut_mean"] = self.cfg.stats["fut_mean"].cuda()
        self.cfg.stats["fut_std"] = self.cfg.stats["fut_std"].cuda()
        return self.p_losses(x, log_dict)