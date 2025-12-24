import logging
import os 
import copy
import math
import os
import pickle
import numpy as np 
import matplotlib.pyplot as plt

from glob import glob
from pathlib import Path

import torch
import torch.nn as nn

from einops import rearrange, reduce

from accelerate import Accelerator
from ema_pytorch import EMA

from tqdm.auto import tqdm

from utils.utils import set_random_seed
from utils.normalization import unnormalize_min_max, unnormalize_sqrt, unnormalize_mean_std
from collections import defaultdict
import torch.nn.functional as F


# helpers functions
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def build_scheduler(optimizer, opt_cfg, total_iters_each_epoch):
    total_epochs = opt_cfg.NUM_EPOCHS
    decay_steps = [x * total_iters_each_epoch for x in opt_cfg.get('DECAY_STEP_LIST', [5, 10, 15, 20])]
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * opt_cfg.LR_DECAY
        return max(cur_decay, opt_cfg.LR_CLIP / opt_cfg.LR)

    if opt_cfg.get('SCHEDULER', None) == 'cosineAnnealingLRwithWarmup':
        # cosine annealing with linear warmup
        total_iterations = total_epochs * total_iters_each_epoch
        warmup_iterations = max(1, int(total_iterations * 0.05))  # 5% of total iterations for warmup
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(opt_cfg.LR_CLIP / opt_cfg.LR, step / warmup_iterations))
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iterations - warmup_iterations, eta_min=opt_cfg.LR_CLIP)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iterations])
    elif opt_cfg.get('SCHEDULER', None) == 'lambdaLR':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd)
    elif opt_cfg.get('SCHEDULER', None) == 'linearLR':
        total_iters = total_iters_each_epoch * total_epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=opt_cfg.LR_CLIP / opt_cfg.LR, total_iters=total_iters)
    elif opt_cfg.get('SCHEDULER', None) == 'stepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt_cfg.DECAY_STEP, gamma=opt_cfg.DECAY_GAMMA)
    elif opt_cfg.get('SCHEDULER', None) == 'cosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=opt_cfg.LR_CLIP)
    else:
        scheduler = None
    return scheduler


def build_optimizer(model, opt_cfg):
    if opt_cfg.OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(
            [each[1] for each in model.named_parameters()],
            lr=opt_cfg.LR, weight_decay=opt_cfg.get('WEIGHT_DECAY', 0)
        )
    elif opt_cfg.OPTIMIZER == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg.LR, weight_decay=opt_cfg.get('WEIGHT_DECAY', 0))
    else:
        assert False
    return optimizer


class Trainer(object):
    def __init__(
        self,
		cfg,
		denoiser, 
		train_loader, 
		test_loader, 
        val_loader=None,
		tb_log=None,
		logger=None,
        gradient_accumulate_every=1,
		ema_decay=0.995,
		ema_update_every=1,
        save_samples=False,
        *awgs, **kwargs
    ):
        super().__init__()

        # init
        self.cfg = cfg
        self.denoiser = denoiser
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = default(val_loader, test_loader)
        self.tb_log = tb_log
        self.logger = logger

        self.gradient_accumulate_every = gradient_accumulate_every
        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every
        
        # config fields
        if cfg.denoising_method == 'fm':
            self.denoising_steps = cfg.sampling_steps
            self.denoising_schedule = cfg.t_schedule 
        else:
            raise NotImplementedError(f'Denoising method [{cfg.denoising_method}] is not implemented yet.')
        
        self.save_dir = Path(cfg.cfg_dir)

        # sampling and training hyperparameters
        self.save_and_sample_every = cfg.checkpt_freq * len(train_loader)
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = cfg.OPTIMIZATION.NUM_EPOCHS * len(train_loader)

        self.save_samples = save_samples
        
        # accelerator
        self.accelerator = Accelerator(
            split_batches = True,
            mixed_precision = 'no'
        )

        # EMA model
        if self.accelerator.is_main_process:
            self.ema = EMA(denoiser, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        # optimizer
        self.opt = build_optimizer(self.denoiser, self.cfg.OPTIMIZATION)
        self.scheduler = build_scheduler(self.opt, self.cfg.OPTIMIZATION, len(self.train_loader))

        # prepare model, dataloader, optimizer with accelerator
        self.denoiser, self.opt = self.accelerator.prepare(self.denoiser, self.opt)

        # datasets and dataloaders
        train_dl_ = self.accelerator.prepare(train_loader)
        self.train_loader = train_dl_
        self.dl = cycle(train_dl_)

        self.test_loader = self.accelerator.prepare(test_loader)

        val_loader = default(val_loader, test_loader)
        self.val_loader = self.accelerator.prepare(val_loader)

        # set counters and training states
        self.step = 0
        self.best_ade_min = float('inf')

        if self.cfg.get('data_norm', None) == 'sqrt':
            self.sqrt_a_ = torch.tensor([self.cfg.sqrt_x_a, self.cfg.sqrt_y_a], device=self.device)
            self.sqrt_b_ = torch.tensor([self.cfg.sqrt_x_b, self.cfg.sqrt_y_b], device=self.device)

        # print the number of model parameters
        self.print_model_params(self.denoiser, 'Stage One Model')

    def print_model_params(self, model: nn.Module, name: str):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"[{name}] Trainable/Total: {trainable_num}/{total_num}")

    @property
    def device(self):
        return self.cfg.device

    def save_ckpt(self, ckpt_name):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.denoiser),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }
        torch.save(data, os.path.join(self.cfg.model_dir, f'{ckpt_name}.pt'))

    def save_last_ckpt(self):
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.denoiser),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        torch.save(data, os.path.join(self.cfg.model_dir, 'checkpoint_last.pt'))
    
    def load(self, ckpt_name):
        accelerator = self.accelerator

        data = torch.load(os.path.join(self.cfg.model_dir, f'{ckpt_name}.pt'), map_location=self.device, weights_only=True)

        model = self.accelerator.unwrap_model(self.denoiser)
        model.load_state_dict(data['model'], strict=False)

        self.step = data['step']
        # self.opt.load_state_dict(data['opt'], strict=False)
        if self.accelerator.is_main_process:
            # pass
            self.ema.load_state_dict(data["ema"], strict=False)

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'], strict=False)

    def train(self):
        """
        Training loop
        """

        # init
        accelerator = self.accelerator
        # 取出 HuggingFace Accelerate 的加速器对象（负责混精、分布式、梯度裁剪等）
        self.logger.info('training start')
        # 打印“开始训练”的日志
        iter_per_epoch = self.train_num_steps // self.cfg.OPTIMIZATION.NUM_EPOCHS
        # 依据设定的总训练步数 train_num_steps 与 epoch 数，估算 每个 epoch 的迭代步数（用于记录/命名 checkpoint）

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                # 外层 while：以 “总步数” 为上限的训练循环（不是按 epoch 计，而是按 step 计）
                # init per-iteration variables

                # 统计当前“梯度累积窗口内”的损失总和（便于日志打印）
                total_loss = 0.
                # 切换 教师/当前模型（这里命名为 denoiser）为训练模式
                self.denoiser.train()
                # EMA 模型也切到训练模式（某些层有 train/eval 行为差异时保持一致）
                self.ema.ema_model.train()
                # 梯度累积：在一次优化器 step() 前，累积若干个小批次的梯度，以实现更大的“等效 batch size”。
                for _ in range(self.gradient_accumulate_every):
                    # 从数据迭代器 self.dl 取下一个 batch，并搬到目标设备（GPU/TPU）
                    # 构建一个dict保存数据键值对
                    data = {k : v.to(self.device) for k, v in next(self.dl).items()}
                    # 记录当前处于第几个 epoch（用 step 反推），用于传给模型内部记录或损失项分支控制等。
                    log_dict = {'cur_epoch': self.step // iter_per_epoch}

                    if self.cfg.get('perturb_ctx', 0.0):
                        # used in SDD dataset
                        # 为每个样本生成一个标量缩放 scale_ ~ N(1, σ)，放大或缩小“历史轨迹的原始尺度”，增强鲁棒性/泛化。
                        bs = data['past_traj'].shape[0]
                        scale_ = torch.randn((bs), device=self.device) * self.cfg.perturb_ctx + 1
                        # 只对 past_traj_original_scale 进行缩放，不改变标准化后的张量（通常模型前向用的是标准化值）
                        data['past_traj_original_scale'] = data['past_traj_original_scale'] * scale_[:, None, None, None]

                    # compute the loss
                    # 混合精度前向（由 Accelerate 统一管理）：节省显存、提升速度。
                    with self.accelerator.autocast():
                        # loss：总损失（包含下列项加权和）
                        # loss_reg：回归项（如与 GT 轨迹的 L2/L1 或 IMLE/Chamfer 的“最近邻”回归）
                        # loss_cls：分类/互斥项（K-shot 多模态分支的“哪一条更接近 GT”的分类或分散正则）
                        # loss_vel：速度/平滑正则（鼓励速度场/轨迹的时间一致性）
                        loss, loss_reg, loss_cls, loss_vel = self.denoiser(data, log_dict)
                        # 由于用 梯度累积，把 loss 除以 gradient_accumulate_every，保证累计后等价于一个大 batch
                        loss = loss / self.gradient_accumulate_every
                        # total_loss 用于进度条显示（统计这次累积窗口的损失之和）
                        total_loss += loss.item()
                    # 由 Accelerate 统一实现 混精反传（兼容多卡/TPU/零冗余优化器等）
                    self.accelerator.backward(loss)

                    # log to tensorboard
                    if self.tb_log is not None:
                        self.tb_log.add_scalar('train/loss_total', loss.item(), self.step)
                        self.tb_log.add_scalar('train/loss_reg', loss_reg.item(), self.step)
                        self.tb_log.add_scalar('train/loss_cls', loss_cls.item(), self.step)
                        self.tb_log.add_scalar('train/loss_vel', loss_vel.item(), self.step)
                        self.tb_log.add_scalar('train/learning_rate', self.opt.param_groups[0]["lr"], self.step)
                # 以 self.step 作为横坐标（全局 step）
                pbar.set_description(f'total loss: {total_loss:.4f}, loss_reg: {loss_reg:.4f}, loss_cls: {loss_cls:.4f}, loss_vel: {loss_vel:.4f}, lr: {self.opt.param_groups[0]["lr"]:.6f}')
                # 更新进度条的文本：显示这次梯度累积窗口的损失统计与当前 LR。
                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.denoiser.parameters(), self.cfg.OPTIMIZATION.GRAD_NORM_CLIP)
                # 优化器更新参数，并清空梯度（配合上面的梯度累积，这里只在一个“累积窗口”后调用一次）
                self.opt.step()
                self.opt.zero_grad()
                # 再次同步，确保更新后的参数在各进程一致。
                accelerator.wait_for_everyone()
                # 仅主进程更新 EMA（Exponential Moving Average） 权重
                # EMA 的模型用于评估与生成，通常更平滑、泛化更好。
                if accelerator.is_main_process:
                    self.ema.update()
                    # checkpt test and save the best validation model
                    if (self.step + 1) >= self.save_and_sample_every and (self.step + 1) % self.save_and_sample_every == 0:
                        fut_traj_gt, performance, n_samples = self.eval_dataloader(testing_mode=False, training_err_check=False)

                        # update the best model
                        if performance['ADE_min'][3] < self.best_ade_min:
                            self.best_ade_min = performance['ADE_min'][3]
                            self.logger.info(f'Current best ADE_MIN: {self.best_ade_min/n_samples}')
                            self.save_ckpt('checkpoint_best')

                        # save the model and remove the old models
                        cur_epoch = self.step // iter_per_epoch

                        ckpt_list = glob(os.path.join(self.cfg.model_dir, 'checkpoint_epoch_*.pt*'))
                        ckpt_list.sort(key=os.path.getmtime)

                        if ckpt_list.__len__() >= self.cfg.max_num_ckpts:
                            for cur_file_idx in range(0, len(ckpt_list) - self.cfg.max_num_ckpts + 1):
                                os.remove(ckpt_list[cur_file_idx])

                        self.save_ckpt('checkpoint_epoch_%d' % cur_epoch)

                self.step += 1
                pbar.update(1)
                self.scheduler.step() 

                # end of one training iteration
            # end of training loop

        self.save_last_ckpt()

        self.logger.info('training complete')
    
    def compute_ADE_FDE(self, distances, end_frame):
        '''
        Helper function to compute ADE and FDE
        distances: [b*num_agents, k_preds, future_frames] or [b*num_agents, timestamps, k_preds, future_frames]
        ade_frames: int
        fde_frame: int
        '''
        # print("dis shape ADE = {}".format(distances.shape))
        ade_best = (distances[..., :end_frame]).mean(dim=-1).min(dim=-1).values.sum(dim=0)
        fde_best = (distances[..., end_frame-1]).min(dim=-1).values.sum(dim=0)
        ade_avg = (distances[..., :end_frame]).mean(dim=-1).mean(dim=-1).sum(dim=0)
        fde_avg = (distances[..., end_frame-1]).mean(dim=-1).sum(dim=0)
        return ade_best, fde_best, ade_avg, fde_avg
    
    ### TODO: add the eval of JADE/JFDE
    ### Based on https://arxiv.org/abs/2305.06292 Joint metric for ADE and FDE
    def compute_JADE_JFDE(self, distances, end_frame):
        '''
        Helper function to compute JADE and JFDE
        distances: [b*num_agents, k_preds, future_frames] or [b*num_agents, timestamps, k_preds, future_frames]
        ade_frames: int
        fde_frame: int
        '''
        jade_best = (distances[..., :end_frame]).mean(dim=-1).sum(dim=0).min(dim=-1).values
        jfde_best = (distances[..., end_frame-1]).sum(dim=0).min(dim=-1).values
        jade_avg = (distances[..., :end_frame]).mean(dim=-1).sum(dim=0).mean(dim=0)
        jfde_avg = (distances[..., end_frame-1]).sum(dim=0).mean(dim=-1)
        return jade_best, jfde_best, jade_avg, jfde_avg

    def compute_avar_fvar(self, pred_trajs, end_frame):
        '''
        Helper function to compute AVar and FVar
        predictions: [b*num_agents,k_preds, future_frames, dim]
        ade_frames: int
        fde_frame: int
        '''
        a_var = pred_trajs[..., :end_frame,:].var(dim=(1,3)).mean(dim=1).sum()
        f_var = pred_trajs[..., end_frame-1,:].var(dim=(1,2)).sum()
        return a_var, f_var

    def compute_MASD(self, pred_trajs, end_frame):
        '''
        Helper function to compute MASD
        predictions: [b*num_agents,k_preds, future_frames, dim]
        ade_frames: int
        fde_frame: int
        '''
        # Reshape for pairwise computation: (B, T, N, D)
        predictions = pred_trajs[:,:,:end_frame,:].permute(0, 2, 1, 3)  # Shape: (B, T, N, D)

        # Compute pairwise L2 distances among N samples at each (B, T)
        pairwise_distances = torch.cdist(predictions, predictions, p=2)  # Shape: (B, T, N, N)

        # Get the maximum squared distance among all pairs (excluding diagonal)
        max_squared_distance = pairwise_distances.max(dim=-1)[0].max(dim=-1)[0]  # Shape: (B, T)

        # Compute the final MASD metric
        masd = max_squared_distance.mean(dim=-1).sum()
        return masd


    @torch.no_grad()
    def test(self, mode, eval_on_train=False):
        # init
        self.logger.info(f'testing start with the {mode} ckpt')

        set_random_seed(42)
        print("final path = {}".format(os.path.join(self.cfg.model_dir, 'checkpoint_last.pt')))
        if mode == 'last':
            ckpt_states = torch.load(os.path.join(self.cfg.model_dir, 'checkpoint_last.pt'), map_location=self.device, weights_only=True)
        else:
            ckpt_states = torch.load(os.path.join(self.cfg.model_dir, 'checkpoint_best.pt'), map_location=self.device, weights_only=True)

        self.denoiser = self.accelerator.unwrap_model(self.denoiser)
        self.denoiser.load_state_dict(ckpt_states['model'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(ckpt_states["ema"])
  
        # testing_mode=False, training_err_check=False
        if eval_on_train:
            fut_traj_gt, _, _ = self.eval_dataloader(training_err_check=True, save_trajs=True)
        else:
            fut_traj_gt, _, _ = self.eval_dataloader(testing_mode=True, save_trajs=True)
        self.logger.info(f'testing complete with the {mode} ckpt')

    @torch.no_grad()
    def tolerance_test(self, mode, eval_on_train=False):
        # init
        self.logger.info(f'tolerance testing start with the {mode} ckpt')

        set_random_seed(42)
        print("final path = {}".format(os.path.join(self.cfg.model_dir, 'checkpoint_last.pt')))
        if mode == 'last':
            ckpt_states = torch.load(os.path.join(self.cfg.model_dir, 'checkpoint_last.pt'), map_location=self.device,
                                     weights_only=True)
        else:
            ckpt_states = torch.load(os.path.join(self.cfg.model_dir, 'checkpoint_best.pt'), map_location=self.device,
                                     weights_only=True)

        self.denoiser = self.accelerator.unwrap_model(self.denoiser)
        self.denoiser.load_state_dict(ckpt_states['model'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(ckpt_states["ema"])

        # predict_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        deltas = [i for i in range(-15, 15, 3)]
        instr_classes = ["L", "F", "R"]
        none_instr_id= 0

        results = self.estimate_temporal_tolerance(
            deltas=deltas,
            instr_classes=instr_classes,
            none_instr_id=none_instr_id,
            training_err_check=True, save_trajs=True
        )
        print("-------------------")
        for item in results.keys():
            print(item, results[item])

        # bin_size = 5  # 或 2 * dt_safe
        # instr_classes = [1, 2, 3]  # F/L/R
        #
        # res_cls = self.estimate_class_tolerance(
        #     bin_size=bin_size,
        #     instr_classes=instr_classes,
        #     none_instr_id=0,
        # )
        # print("-------------------")
        # for item in res_cls.keys():
        #     print(item, res_cls[item])

        self.logger.info(f'testing complete with the {mode} ckpt')


    def sample_from_denoising_model(self, data):
        """
        Return the samples from denoising model in normal scale
        """

        # [B, K, A, T*F], [B, S, K, A, T*F], [B, S, K, A, T*F], [B, K, A]
        pred_traj, pred_traj_at_t, t_seq, y_t_seq, pred_score = self.denoiser.sample(data, num_trajs=self.cfg.denoising_head_preds, return_all_states=self.save_samples)
        # print("??? pred_traj shape = {}".format(pred_traj.shape))
        assert list(pred_traj.shape[2:]) == [self.cfg.agents, self.cfg.MODEL.MODEL_OUT_DIM]

        pred_traj = rearrange(pred_traj, 'b k a (f d) -> (b a) k f d', f=self.cfg.future_frames)[...,0:3]  # [B, k_preds, 11, 40] -> [B * 11, k_preds, 20, 2]
        # print("???? pred_traj shape = {}".format(pred_traj.shape))
        pred_traj_at_t = rearrange(pred_traj_at_t, 'b t k a (f d) -> (b a) t k f d', f=self.cfg.future_frames)[...,0:3]  # [B, k_preds, 11, 40] -> [B * 11, k_preds, 20, 2]

        if self.cfg.get('data_norm', None) == 'min_max':
            # pred_traj = unnormalize_min_max(pred_traj, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1)
            # pred_traj_at_t = unnormalize_min_max(pred_traj_at_t, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1)
            # print("????? pred_traj = {} {}".format(self.cfg.stats["fut_mean"], self.cfg.stats["fut_std"]))
            pred_traj = unnormalize_mean_std(pred_traj, self.cfg.stats["fut_mean"], self.cfg.stats["fut_std"],
                                                     1)  # [B, K, A, T, D] 没还原
            # print("?????? pred_traj shape = {}".format(pred_traj))
            pred_traj_at_t = unnormalize_mean_std(pred_traj_at_t, self.cfg.stats["fut_mean"],
                                                   self.cfg.stats["fut_std"], 1)  # [B, K, A, T, D] 没还原

        elif self.cfg.get('data_norm', None) == 'sqrt':
            pred_traj = unnormalize_sqrt(pred_traj, self.sqrt_a_, self.sqrt_b_)
            pred_traj_at_t = unnormalize_sqrt(pred_traj_at_t, self.sqrt_a_, self.sqrt_b_)
        elif self.cfg.get('data_norm', None) == 'hist10pred20':
            pass
        else:
            raise NotImplementedError(f'Data normalization [{self.cfg.data_norm}] is not implemented yet.')

        return pred_traj, pred_traj_at_t, t_seq, y_t_seq, pred_score
    

    def save_latent_states(self, t_seq_ls, y_t_seq_ls, y_pred_data_ls, x_data_ls, pred_score_ls, file_name):
        self.logger.info("Begin to save the denoising samples...")

        if self.cfg.dataset in ['nba', 'sdd', 'eth_ucy', 'rat']:
            keys_to_save = ['past_traj', 'fut_traj', 'past_traj_original_scale', 'fut_traj_original_scale', 'fut_traj_vel']
        else:
            raise NotImplementedError(f'Dataset [{self.cfg.dataset}] is not implemented yet.')
    
        states_to_save = {k: [] for k in keys_to_save}

        states_to_save['t'] = []
        states_to_save['y_t'] = []
        states_to_save['y_pred_data'] = []
        # states_to_save['pred_score'] = []

        for i_batch, (t_seq, y_t_seq, y_pred_data, x_data) in enumerate(zip(t_seq_ls, y_t_seq_ls, y_pred_data_ls, x_data_ls)):
            try:
                t = t_seq.detach().cpu().numpy().reshape(1, -1)
            except:
                breakpoint()
            states_to_save['t'].append(t)

            y_t_seq = y_t_seq.detach().cpu().numpy()
            states_to_save['y_t'].append(y_t_seq)

            y_pred_data = y_pred_data.detach().cpu().numpy()
            states_to_save['y_pred_data'].append(y_pred_data)

            # pred_score = pred_score.detach().cpu().numpy()
            # states_to_save['pred_score'].append(pred_score)

            for key in keys_to_save:
                x_data_val_ = x_data[key].detach().cpu().numpy()
                assert len(y_t_seq) == len(x_data_val_)
                states_to_save[key].append(x_data_val_)

        for key in states_to_save:
            states_to_save[key] = np.concatenate(states_to_save[key], axis=0)

        # clean up the cfg and remove any path related fields
        cfg_ = copy.deepcopy(self.cfg.yml_dict)

        def _remove_path_fields(cfg):
            for k in list(cfg.keys()):
                if 'path' in k or 'dir' in k:
                    cfg.pop(k)
                elif isinstance(cfg[k], dict):
                    _remove_path_fields(cfg[k])
                else:
                    try:
                        if os.path.isdir(cfg[k]) or os.path.isfile(cfg[k]):
                            cfg.pop(k)
                    except:
                        pass

        _remove_path_fields(cfg_)

        num_datapoints = len(states_to_save['y_t'])
        meta_data = {'cfg': cfg_, 'size': num_datapoints}

        states_to_save['meta_data'] = meta_data
        
        # save_path = os.path.join(self.cfg.sample_dir, f'{file_name}.npz')
        # np.savez_compressed(save_path, **states_to_save)

        save_path = os.path.join(self.cfg.sample_dir, f'{file_name}.pkl')
        self.logger.info("Saving the denoising samples to {}".format(save_path))
        pickle.dump(states_to_save, open(save_path, 'wb'))

    
    def eval_dataloader(self, testing_mode=False, training_err_check=False, save_trajs=False):
        """
        General API to evaluate the dataloader/dataset
        """
        ### turn on the eval mode
        self.denoiser.eval()   
        self.ema.ema_model.eval()
        self.logger.info(f'Record the statistics of samples from the denoising model')

        if testing_mode:
            self.logger.info(f'Start recording test set ADE/FDE...')
            status = 'test'
            dl = self.test_loader
        elif training_err_check:
            self.logger.info(f'Start recording training set ADE/FDE...')
            status = 'train'
            dl = self.train_loader
        else:
            self.logger.info(f'Start recording validation set ADE/FDE...')
            status = 'val'
            dl = self.val_loader
      
        ### setup the performance dict
        performance = {'FDE_min': [0,0,0,0,0,0], 'ADE_min': [0,0,0,0,0,0], 'FDE_avg': [0,0,0,0,0,0], 'ADE_avg': [0,0,0,0,0,0], 'A_var': [0,0,0,0,0,0], 'F_var': [0,0,0,0,0,0], 'MASD': [0,0,0,0,0,0]}
        performance_joint = {'JFDE_min': [0,0,0,0,0,0], 'JADE_min': [0,0,0,0,0,0], 'JFDE_avg': [0,0,0,0,0,0], 'JADE_avg': [0,0,0,0,0,0]}
        num_trajs = 0
        t_seq_ls, y_t_seq_ls, y_pred_data_ls, x_data_ls = [], [], [], []
        ### record running time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        pred_trajs = []
        hits_trajs = []
        hist_cond_cue = []
        fut_cond_cue = []
        fut_trajs = []

        for i_batch, data in enumerate(dl): 
            bs = int(data['batch_size'])
            data = {k : v.to(self.device) for k, v in data.items()}

            pred_traj, pred_traj_t, t_seq, y_t_seq, pred_score = self.sample_from_denoising_model(data)

            pred_trajs.append(pred_traj)
            hits_trajs.append(data["past_traj_original_scale"])
            hist_cond_cue.append(data["hist_cond_cue"])
            fut_cond_cue.append(data["fut_cond_cue"])
            fut_trajs.append(data['fut_traj'])

            # 没还原对比没还原
            fut_traj = rearrange(data['fut_traj'], 'b a f d -> (b a) f d')               # [B, A, T, F] -> [B * A, T, F]
            fut_traj_gt = fut_traj.unsqueeze(1).repeat(1, self.cfg.denoising_head_preds, 1, 1)          # [B * A, K, T, F]
            distances = (fut_traj_gt - pred_traj).norm(p=2, dim=-1)                                     # [B * A, K, T]
            distances_t = (pred_traj_t - fut_traj_gt.unsqueeze(1)).norm(p=2, dim=-1)                    # [B * A, S, K, T]
            
            ade_fde_ = self.compute_ADE_FDE(distances_t, self.cfg.future_frames)                        # 4 * [S], denoising steps

            if self.cfg.dataset == 'nba':
                freq = 5 
                factor_time = 1
            elif self.cfg.dataset == 'eth_ucy':
                freq = 3
                factor_time = 1.2
            elif self.cfg.dataset == 'sdd':
                freq = 3
                factor_time = 1.2
            elif self.cfg.dataset in ["rat", "babel"]:
                freq = 5
                factor_time = 0.3

            for time in range(1, 7):
                ade, fde, ade_avg, fde_avg = self.compute_ADE_FDE(distances, int(time * freq))
                jade, jfde, jade_avg, jfde_avg = self.compute_JADE_JFDE(distances, int(time * freq)) 
                a_var, f_var = self.compute_avar_fvar(pred_traj, int(time * freq))
                masd = self.compute_MASD(pred_traj, int(time * freq))
                performance_joint['JADE_min'][time - 1] += jade.item()
                performance_joint['JFDE_min'][time - 1] += jfde.item()
                performance_joint['JADE_avg'][time - 1] += jade_avg.item()
                performance_joint['JFDE_avg'][time - 1] += jfde_avg.item()
                performance['ADE_min'][time - 1] += ade.item()
                performance['FDE_min'][time - 1] += fde.item()
                performance['ADE_avg'][time - 1] += ade_avg.item()
                performance['FDE_avg'][time - 1] += fde_avg.item()
                performance['A_var'][time - 1] += a_var.item()
                performance['F_var'][time - 1] += f_var.item()
                performance['MASD'][time - 1] += masd.item()

            assert freq * 6 == self.cfg.future_frames, 'Freq {} and number of frames {} do not match'.format(freq, self.cfg.future_frames)
             
            num_trajs += fut_traj.shape[0]

            # save the denoising samples
            if self.save_samples:
                cutoff_timesteps = 5  # only save the last 5 timesteps sampling latents to reduce the storage size

                y_t_seq = y_t_seq[:, -cutoff_timesteps:]
                y_t_seq = rearrange(y_t_seq, 'b s k a (f d) -> b s k a f d', f=self.cfg.future_frames)

                pred_traj = rearrange(pred_traj, '(b a) k f d -> b k a f d', b=bs)  # [B, K, A, F, D]
            
                num_datapoints = len(y_t_seq)

                t_seq_ls = [t_seq]
                y_t_seq_ls = [y_t_seq]
                y_pred_data_ls = [pred_traj]
                x_data_ls = [data]
                pred_score_ls = [pred_score]

                solver_tag = self.cfg.get('solver_tag', '')
                save_name = f'denoising_samples_{status}_batch_{i_batch}_{num_datapoints}_{solver_tag}'
                self.save_latent_states(t_seq_ls, y_t_seq_ls, y_pred_data_ls, x_data_ls, pred_score_ls, save_name)
                
                t_seq_ls, y_t_seq_ls, y_pred_data_ls, x_data_ls, pred_score_ls = [], [], [], [], []
                
        end.record()
        torch.cuda.synchronize()
        self.logger.info(f'Total runtime: {start.elapsed_time(end):5f} ms')
        self.logger.info(f'Runtime per scene: {start.elapsed_time(end)/len(dl.dataset):5f} ms')
        self.logger.info(f'Number of scenes: {dl.dataset}')
        cur_epoch = self.step // (self.train_num_steps // self.cfg.OPTIMIZATION.NUM_EPOCHS)
        if not testing_mode: 
            self.logger.info(f'{self.step}/{self.train_num_steps}, running inference on {num_trajs} agents (trajectories)')
            for time in range(6):
                if self.tb_log:
                    self.tb_log.add_scalar(f'eval_{status}/ADE_min_{time+1}s', performance['ADE_min'][time]/num_trajs, cur_epoch)
                    self.tb_log.add_scalar(f'eval_{status}/FDE_min_{time+1}s', performance['FDE_min'][time]/num_trajs, cur_epoch)
                    self.tb_log.add_scalar(f'eval_{status}/ADE_avg_{time+1}s', performance['ADE_avg'][time]/num_trajs, cur_epoch)
                    self.tb_log.add_scalar(f'eval_{status}/FDE_avg_{time+1}s', performance['FDE_avg'][time]/num_trajs, cur_epoch)
                    self.tb_log.add_scalar(f'eval_{status}/JADE_min_{time+1}s', performance_joint['JADE_min'][time]/num_trajs, cur_epoch)
                    self.tb_log.add_scalar(f'eval_{status}/JFDE_min_{time+1}s', performance_joint['JFDE_min'][time]/num_trajs, cur_epoch)
                    self.tb_log.add_scalar(f'eval_{status}/JADE_avg_{time+1}s', performance_joint['JADE_avg'][time]/num_trajs, cur_epoch)
                    self.tb_log.add_scalar(f'eval_{status}/JFDE_avg_{time+1}s', performance_joint['JFDE_avg'][time]/num_trajs, cur_epoch)

        # print out the performance
        for time in range(6):
            self.logger.info('--ADE_min({:.1f}s): {:.7f}\t--FDE_min({:.1f}s): {:.7f}'.format(
                (time+1)*factor_time, performance['ADE_min'][time]/num_trajs, time+1, performance['FDE_min'][time]/num_trajs))

      
        for time in range(6):
            self.logger.info('--ADE_avg({:.1f}s): {:.7f}\t--FDE_avg({:.1f}s): {:.7f}'.format(
                time+1, performance['ADE_avg'][time]/num_trajs, time+1, performance['FDE_avg'][time]/num_trajs))

        for time in range(6):
            self.logger.info('--AVar({:.1f}s): {:.7f}\t--FVar({:.1f}s): {:.7f}'.format(
                time+1, performance['A_var'][time]/num_trajs, time+1, performance['F_var'][time]/num_trajs))
        
        for time in range(6):
            self.logger.info('--MASD({:.1f}s): {:.7f}'.format(
                time+1, performance['MASD'][time]/num_trajs))
            
        # print out the joint performance
        for time in range(6):
            self.logger.info('--JADE_min({:.1f}s): {:.7f}\t--JFDE_min({:.1f}s): {:.7f}'.format(
                time+1, performance_joint['JADE_min'][time]/num_trajs, time+1, performance_joint['JFDE_min'][time]/num_trajs))
        
        for time in range(6):
            self.logger.info('--JADE_avg({:.1f}s): {:.7f}\t--JFDE_avg({:.1f}s): {:.7f}'.format(
                time+1, performance_joint['JADE_avg'][time]/num_trajs, time+1, performance_joint['JFDE_avg'][time]/num_trajs))

        if save_trajs:
            pred_trajs_np = []
            for item in pred_trajs:
                item = rearrange(item, '(b a) k f d -> b k a f d', a=8)  # [B, K, A, F, D]
                item = item.cpu()
                item = unnormalize_mean_std(item, self.cfg.stats["fut_mean"], self.cfg.stats["fut_std"],0)  # [B, K, A, T, D]
                item = item.detach().numpy()
                pred_trajs_np.append(item)

            # print(pred_trajs_np[0].shape)
            arr = np.concatenate(pred_trajs_np, axis=0)  # 形状变为 (N, T, 2)
            # print(arr.shape)
            np.save(r"D:\04_code\MoFlow\visualize\trajs\pred_trajs.npy", arr)

            hits_trajs = [item.cpu().detach().numpy() for item in hits_trajs]
            # print(hits_trajs[0].shape)
            arr = np.concatenate(hits_trajs, axis=0)  # 形状变为 (N, T, 2)
            # print(arr.shape)
            np.save(r"D:\04_code\MoFlow\visualize\trajs\hist_trajs.npy", arr)

            hist_cond_cue = [item.cpu().detach().numpy() for item in hist_cond_cue]
            # print(cue_trajs[0].shape)
            arr = np.concatenate(hist_cond_cue, axis=0)  # 形状变为 (N, T, 2)
            # print(arr.shape)
            np.save(r"D:\04_code\MoFlow\visualize\trajs\hist_cue_trajs.npy", arr)

            fut_cond_cue = [item.cpu().detach().numpy() for item in fut_cond_cue]
            # print(cue_trajs[0].shape)
            arr = np.concatenate(fut_cond_cue, axis=0)  # 形状变为 (N, T, 2)
            # print(arr.shape)
            np.save(r"D:\04_code\MoFlow\visualize\trajs\fut_cue_trajs.npy", arr)

            # fut_trajs = [item.cpu().detach().numpy() for item in fut_gt_trajs]
            fut_trajs_np = []
            for item in fut_trajs:
                # item = rearrange(item, '(b a) f d -> b a f d', a=8)  # [B, K, A, F, D]
                item = item.cpu()
                item = unnormalize_mean_std(item, self.cfg.stats["fut_mean"], self.cfg.stats["fut_std"],0)  # [B, K, A, T, D]
                item = item.detach().numpy()
                fut_trajs_np.append(item)

            arr = np.concatenate(fut_trajs_np, axis=0)  # 形状变为 (N, T, 2)
            print(arr.shape)
            np.save(r"D:\04_code\MoFlow\visualize\trajs\fut_gt_trajs.npy", arr)

        return fut_traj_gt, performance, num_trajs

    @torch.no_grad()
    def estimate_temporal_tolerance(
            self,
            deltas,
            instr_classes,
            none_instr_id: int = 0,
            device: str = "cuda",
            testing_mode=False, training_err_check=False, save_trajs=False
    ):
        """
        估计时间容错曲线 ΔADE_u(δ)：
        - 对每个验证样本中的指令事件 (n, tau, u_tau)
        - 将该事件偏移 delta 帧（tau+delta)，重新预测未来，统计 ADE 增量
        - 按指令类别 u 和 delta 做平均

        参数：
        - predict_fn: (hist, cmd) -> pred_fut
        - val_loader: 迭代器，返回 batch dict
        - deltas: 例如 [-5, -3, -1, 1, 3, 5]，可包含正负数
        - hist_len: 历史长度 Th（未来段从 hist_len 开始）
        - instr_classes: 需要统计的指令类别列表，例如 [1,2,3] (F/L/R)
        - none_instr_id: 表示“无指令”的 id（通常为 0）
        - max_events_per_batch: 每个 batch 最多抽多少个指令事件做扰动（控制计算量）
        - device: 设备
        - hist_key, fut_key, cmd_key: batch 中的字段名

        返回：
        - results: dict 包含
            - 'deltas': list[int]
            - 'class_ids': list[int]
            - 'mean_delta_ade': np.ndarray [C, D]  (C=类别数, D=|deltas|)
            - 'counts': np.ndarray [C, D]
        """
        # # turn on the eval mode
        self.denoiser.eval()
        self.ema.ema_model.eval()
        self.logger.info(f'Estimated time tolerance curve')

        # 加载数据集
        self.logger.info(f'Start recording validation set ADE/FDE...')
        status = 'val'
        dl = self.val_loader

        # 统计指令类别
        instr_classes = list(instr_classes)
        num_classes = len(instr_classes)
        deltas = list(deltas)
        num_deltas = len(deltas)

        # 类别 id -> 索引
        class2idx = {c: i+1 for i, c in enumerate(instr_classes)}
        delta2idx = {d: i for i, d in enumerate(deltas)}

        # 累计器
        sum_delta_ade = torch.zeros(num_classes, num_deltas, device=device)
        cnt_delta_ade = torch.zeros(num_classes, num_deltas, device=device)

        def compute_agent_ADE_minK(pred_traj, fut_traj):
            """
            pred_traj: [BA, K, T, D]
            fut_traj:  [BA,    T, D]
            返回：ade_min: [BA]
            """
            # [BA, K, T, D] - [BA, 1, T, D] -> [BA, K, T, D]
            diff = pred_traj - fut_traj.unsqueeze(1)
            # L2 over coord
            l2 = torch.norm(diff, dim=-1)  # [BA, K, T]
            ade_k = l2.mean(dim=-1)  # [BA, K]
            ade_min, _ = ade_k.min(dim=1)  # [BA]
            return ade_min

        def recompute_time_since_last_cmd(fut_cond):
            # fut_cond: [B, T, C]
            B, T, C = fut_cond.shape
            fut_cond = fut_cond.clone()
            for b in range(B):
                last_cmd_step = None
                for t in range(T):
                    # 判定当前步是否有“有效指令”（非“无指令”类）
                    cmd_cls = fut_cond[b, t, :4].argmax().item()
                    if cmd_cls != 0:  # 0 类表示“无指令”
                        last_cmd_step = t
                    if last_cmd_step is None:
                        fut_cond[b, t, 6] = 0.0
                    else:
                        fut_cond[b, t, 6] = float(t - last_cmd_step)
            return fut_cond

        for i_batch, data in enumerate(dl):
            # 取数据并搬到 device
            B = int(data['batch_size'])
            fut_cond_cue = data["fut_cond_cue"]

            data = {k: v.to(self.device) for k, v in data.items()}

            Th = self.cfg.future_frames
            Tf = self.cfg.past_frames
            T_total = Th + Tf

            # 基线预测（真实指令，不偏移）
            # base_pred = predict_fn(hist, cmd)  # [B,Tf,...]
            pred_traj, pred_traj_t, t_seq, y_t_seq, pred_score = self.sample_from_denoising_model(data)

            fut_traj = rearrange(data['fut_traj'], 'b a f d -> (b a) f d')  # [B, A, T, F] -> [B * A, T, F]
            fut_traj_gt = fut_traj.unsqueeze(1).repeat(1, self.cfg.denoising_head_preds, 1, 1)  # [B * A, K, T, F]
            print("fut_traj_gt shape = {}".format(fut_traj_gt.shape))
            print("pred_traj shape = {}".format(pred_traj.shape))
            distances = (fut_traj_gt - pred_traj).norm(p=2, dim=-1)                                     # [B * A, K, T]
            # distances_t = (pred_traj_t - fut_traj_gt.unsqueeze(1)).norm(p=2, dim=-1)  # [B * A, S, K, T]
            # ade_base,_,_,_ = self.compute_ADE_FDE(distances, self.cfg.future_frames)       # 4 * [S], denoising steps
            ade_base = compute_agent_ADE_minK(pred_traj, fut_traj)  # [BA]
            print("ade_base shape = {}".format(ade_base))

            # 在未来段内寻找所有“有指令”的事件 (n, tau)
            # tau: 相对于整体 cmd 序列的 index，范围 [Th, Th+Tf-1]
            events = []  # list of (n, tau, instr_id)
            for n in range(B):
                # 提取未来部分的 cmd
                fut_cmd = fut_cond_cue[n, ...]  # [Tf]
                print("fut_cmd shape = {}".format(fut_cmd.shape))
                # print("fut_cmd = {}".format(fut_cmd))
                # 找到非 none_instr_id 的位置
                idx = (fut_cmd[:, none_instr_id] != 1).nonzero(as_tuple=False).view(-1)
                print("idx = {}".format(idx))
                for tau in idx.tolist():
                    print("one hot = {}".format(fut_cond_cue[n, tau, :4]))
                    instr_id = int(fut_cond_cue[n, tau, :4].argmax(dim=-1))
                    print("instr_id = {}".format(instr_id))
                    print("class2idx = {}".format(class2idx))
                    if instr_id in class2idx.values():
                        # 增加一个事件，格式为batch内的编号n，相对时间戳tau，指令类别instr_id
                        events.append((n, tau, instr_id))

            # 1. 按指令类别把 events 分组：instr_id -> list[(n, tau)]
            events_by_class = defaultdict(list)
            for (n, tau, instr_id) in events:
                events_by_class[instr_id].append((n, tau))

            device = fut_cond_cue.device
            dtype = fut_cond_cue.dtype
            onehot_empty = torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=dtype, device=device)

            # 2. 遍历每一类指令 & 每个 delta
            for instr_id, ev_list in events_by_class.items():
                class_idx = instr_id  # 你的代码里就是用这个作为类别
                print("instr_id = {} ev_list = {}".format(instr_id, ev_list))
                if len(ev_list) == 0:
                    continue

                # 把这一类指令对应的所有 n, tau 拉成向量，方便批量索引
                n_list = torch.tensor([e[0] for e in ev_list], dtype=torch.long, device=device)
                tau_list = torch.tensor([e[1] for e in ev_list], dtype=torch.long, device=device)

                # 对应的 baseline ADE，一次性取出来
                base_ade_vec = ade_base[n_list]  # shape [Ne], Ne = 当前类别的 event 数

                for d in deltas:
                    delta_idx = delta2idx[d]

                    # 计算偏移后的 new_tau，并筛掉非法的
                    new_tau_list = tau_list + d
                    valid_mask = (new_tau_list >= 0) & (new_tau_list < Tf)
                    if not valid_mask.any():
                        continue

                    n_valid = n_list[valid_mask]  # [Nv]
                    tau_valid = tau_list[valid_mask]  # [Nv]
                    new_tau_valid = new_tau_list[valid_mask]  # [Nv]
                    base_ade_valid = base_ade_vec[valid_mask]

                    # 3. 构造这一类 + 这个 delta 的整体扰动序列
                    fut_cond_perturb = fut_cond_cue.clone()  # [N, T, C] 或 [B*A, T, C]

                    # (1) 把原 tau 的指令复制到 new_tau 的位置
                    fut_cond_perturb[n_valid, new_tau_valid, :] = fut_cond_cue[n_valid, tau_valid, :]

                    # (2) 原 tau 位置清空为 "无指令"（onehot_empty）
                    fut_cond_perturb[n_valid, tau_valid, :] = onehot_empty
                    recompute_time_since_last_cmd(fut_cond_perturb)

                    # 4. 一次性评估这一类所有 event 在该 delta 下的影响
                    data_perturb = dict(data)
                    data_perturb['fut_cond_cue'] = fut_cond_perturb

                    pred_traj_perturb, _, _, _, _ = self.sample_from_denoising_model(data_perturb)
                    ade_perturb = compute_agent_ADE_minK(pred_traj_perturb, fut_traj)  # [N]，与 ade_base 对齐

                    ade_perturb_valid = ade_perturb[n_valid]  # 只取当前类、当前 delta 里有效的那些 n
                    delta_ades = ade_perturb_valid - base_ade_valid  # [Nv]

                    # 5. 把这一类 + 这个 delta 下所有 event 的贡献累加进统计量
                    sum_delta_ade[class_idx - 1, delta_idx] += delta_ades.sum().item()
                    cnt_delta_ade[class_idx - 1, delta_idx] += float(delta_ades.numel())

        if (i_batch + 1) % 10 == 0:
            self.logger.info(f"[TemporalTol] processed batch {i_batch+1}/{len(dl)} "
                             f"(acc events: {int(cnt_delta_ade.sum().item())})")

        # 汇总结果
        mean_delta_ade = torch.zeros_like(sum_delta_ade)
        mask = cnt_delta_ade > 0
        mean_delta_ade[mask] = sum_delta_ade[mask] / cnt_delta_ade[mask]

        results = {
            "deltas": deltas,
            "class_ids": instr_classes,
            "mean_delta_ade": mean_delta_ade.detach().cpu().numpy(),  # [C,D]
            "counts": cnt_delta_ade.detach().cpu().numpy(),  # [C,D]
        }
        return results

    @torch.no_grad()
    def estimate_class_tolerance(
            self,
            bin_size: int,
            instr_classes,
            none_instr_id: int = 0,
            device: str = "cuda",
            testing_mode=False, training_err_check=False, save_trajs=False
    ):
        """
        第二阶段：类别容错统计 ΔADE_b(u -> u')。

        思路：
        - 用 bin_size (≈ δt_safe 或 2δt_safe) 将未来时间 [0, T_fut-1] 划分为若干个 bin
        - 对每个指令事件 (n, tau, u_tau)，确定所属 bin: b = floor(tau / bin_size)
        - 在该 bin 内，把所有类别为 u 的事件一次性替换为 u'，重新预测未来轨迹
        - 统计 ADE 增量，并按 (bin, u, u') 求平均

        参数：
        - bin_size: 每个 bin 的长度（帧数）
        - instr_classes: 需要统计的“有效指令”类别列表，如 [1,2,3] (F/L/R)
        - none_instr_id: “无指令”类别 id，一般为 0
        - device: 设备

        返回：
        results: dict
            - 'bin_size': int
            - 'bin_ranges': List[(start_t, end_t)] 每个 bin 覆盖的时间范围（闭区间）
            - 'class_ids': list[int] 与 instr_classes 一致
            - 'mean_delta_ade': np.ndarray [num_bins, C, C]
            - 'counts': np.ndarray [num_bins, C, C]
        """

        # turn on eval
        self.denoiser.eval()
        self.ema.ema_model.eval()
        self.logger.info(f'[ClassTol] Estimate class tolerance ΔADE_b(u → u\')')

        # 加载验证集
        self.logger.info(f'[ClassTol] Start recording validation set ADE/FDE on val set...')
        status = 'val'
        dl = self.val_loader

        # 类别 / 索引映射（保持和时间容错阶段风格一致）
        instr_classes = list(instr_classes)
        num_classes = len(instr_classes)
        class2idx = {c: i + 1 for i, c in enumerate(instr_classes)}  # 类别id -> 1..C

        # 先初始化“全局”统计容器，需要知道 num_bins，所以我们先等第一批数据出来
        sum_delta_ade = None  # shape [num_bins, C, C]
        cnt_delta_ade = None  # same

        def compute_agent_ADE_minK(pred_traj, fut_traj):
            """
            pred_traj: [BA, K, T, D]
            fut_traj:  [BA,    T, D]
            返回：ade_min: [BA]
            """
            diff = pred_traj - fut_traj.unsqueeze(1)  # [BA, K, T, D]
            l2 = torch.norm(diff, dim=-1)  # [BA, K, T]
            ade_k = l2.mean(dim=-1)  # [BA, K]
            ade_min, _ = ade_k.min(dim=1)  # [BA]
            return ade_min

        # ------------------------ 遍历 batch ------------------------
        total_events = 0
        for i_batch, data in enumerate(dl):
            B = int(data['batch_size'])
            # fut_cond_cue = data["fut_cond_cue"]  # [B, T_fut, C]
            # fut_cond_cue = fut_cond_cue.to(self.device)

            # 其余字段搬到 device
            data = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                    for k, v in data.items()}
            fut_cond_cue = data['fut_cond_cue']

            # 未来轨迹长度 & bin 数量
            T_fut = fut_cond_cue.shape[1]
            num_bins_batch = (T_fut + bin_size - 1) // bin_size

            # 初始化全局统计容器（只需要一次）
            if sum_delta_ade is None:
                sum_delta_ade = torch.zeros(
                    num_bins_batch, num_classes, num_classes, device=device
                )
                cnt_delta_ade = torch.zeros(
                    num_bins_batch, num_classes, num_classes, device=device
                )
                num_bins = num_bins_batch
            else:
                assert num_bins_batch == sum_delta_ade.size(0), \
                    "所有 batch 的未来长度应该相同，以保证 num_bins 一致"

            # ---------- 1. 基线预测 ----------
            pred_traj, pred_traj_t, t_seq, y_t_seq, pred_score = \
                self.sample_from_denoising_model(data)

            fut_traj = rearrange(data['fut_traj'], 'b a f d -> (b a) f d')
            fut_traj_gt = fut_traj  # [BA, T, D]
            ade_base = compute_agent_ADE_minK(pred_traj, fut_traj_gt)  # [BA]，与 batch 内 n 索引对齐（B == BA）

            # ---------- 2. 收集事件：所有“有指令”的 (n, tau, instr_id, bin_id) ----------
            events_by_bin_class = defaultdict(list)  # (bin_id, instr_id) -> [(n, tau), ...]

            for n in range(B):
                fut_cmd = fut_cond_cue[n, ...]  # [T_fut, C]
                # 非 none_instr 的位置
                idx = (fut_cmd[:, none_instr_id] != 1).nonzero(as_tuple=False).view(-1)
                for tau in idx.tolist():  # tau: 0..T_fut-1
                    instr_id = int(fut_cmd[tau, :4].argmax(dim=-1))
                    if instr_id not in class2idx.values():
                        continue
                    bin_id = tau // bin_size
                    events_by_bin_class[(bin_id, instr_id)].append((n, tau))
                    total_events += 1

            if (i_batch + 1) % 10 == 0:
                self.logger.info(f"[ClassTol] collected events: "
                                 f"batch {i_batch + 1}/{len(dl)}, total_events={total_events}")

            # 若没有事件，跳过
            if len(events_by_bin_class) == 0:
                continue

            # ---------- 3. 遍历每个 (bin, 原始类别 u) 组合 ----------
            device = fut_cond_cue.device
            dtype = fut_cond_cue.dtype

            for (bin_id, instr_id), ev_list in events_by_bin_class.items():
                if len(ev_list) == 0:
                    continue

                class_idx = instr_id  # 1..C
                c_from = class_idx - 1

                # 当前 bin + 当前原始类别的所有 (n, tau)
                n_list = torch.tensor([e[0] for e in ev_list], dtype=torch.long, device=device)
                tau_list = torch.tensor([e[1] for e in ev_list], dtype=torch.long, device=device)

                base_ade_vec = ade_base[n_list]  # [Ne]

                # ---------- 4. 对每个目标类别 u' 做一次整体替换 ----------
                for target_id in class2idx.values():  # 遍历 F/L/R 等
                    if target_id == instr_id:
                        continue  # 不需要替换成自身

                    c_to = target_id - 1

                    # 构造 one-hot 目标类别（只改前 4 维）
                    onehot_target = F.one_hot(
                        torch.tensor(target_id, device=device),
                        num_classes=4
                    ).float()  # [4]

                    # (1) 构造扰动后的 fut_cond_cue 副本
                    fut_cond_perturb = fut_cond_cue.clone()  # [B, T_fut, C]
                    fut_cond_perturb[n_list, tau_list, :4] = onehot_target  # 同一时刻改类别

                    # (2) 重新预测
                    data_perturb = dict(data)
                    data_perturb['fut_cond_cue'] = fut_cond_perturb

                    pred_traj_perturb, _, _, _, _ = \
                        self.sample_from_denoising_model(data_perturb)
                    ade_perturb = compute_agent_ADE_minK(
                        pred_traj_perturb, fut_traj_gt
                    )  # [BA]

                    ade_perturb_vec = ade_perturb[n_list]  # [Ne]

                    # (3) 事件级 ΔADE，并累加到 (bin_id, u -> u') 上
                    delta_ades = ade_perturb_vec - base_ade_vec  # [Ne]
                    sum_delta_ade[bin_id, c_from, c_to] += delta_ades.sum().item()
                    cnt_delta_ade[bin_id, c_from, c_to] += float(delta_ades.numel())

        # ------------------------ 4. 汇总统计 ------------------------
        mean_delta_ade = torch.zeros_like(sum_delta_ade)
        mask = cnt_delta_ade > 0
        mean_delta_ade[mask] = sum_delta_ade[mask] / cnt_delta_ade[mask]

        # bin 覆盖范围（闭区间），方便后续画图标注
        bin_ranges = []
        for b in range(sum_delta_ade.size(0)):
            start_t = b * bin_size
            end_t = min((b + 1) * bin_size - 1, T_fut - 1)
            bin_ranges.append((int(start_t), int(end_t)))

        results = {
            "bin_size": bin_size,
            "bin_ranges": bin_ranges,  # List[(start_t, end_t)]
            "class_ids": instr_classes,
            "mean_delta_ade": mean_delta_ade.detach().cpu().numpy(),  # [B,C,C]
            "counts": cnt_delta_ade.detach().cpu().numpy(),  # [B,C,C]
        }
        self.logger.info(f"[ClassTol] Done. total_events = {total_events}")
        return results
