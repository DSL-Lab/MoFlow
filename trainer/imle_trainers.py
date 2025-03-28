
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
from utils.normalization import unnormalize_min_max, unnormalize_sqrt

from .denoising_model_trainers import exists, default, identity, has_int_squareroot, cycle, build_optimizer, build_scheduler


class IMLETrainer(object):
    def __init__(
        self,
		cfg,
		imle_generator, 
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
        self.imle_model = imle_generator
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = default(val_loader, test_loader)
        self.tb_log = tb_log
        self.logger = logger

        self.gradient_accumulate_every = gradient_accumulate_every
        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every
        
        # config fields
        self.save_dir = Path(cfg.cfg_dir)

        # sampling and training hyperparameters
        self.save_and_sample_every = cfg.checkpt_freq * len(train_loader)
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = cfg.OPTIMIZATION.NUM_EPOCHS * len(train_loader)

        self.save_samples = save_samples

        assert self.cfg.latent_tau == 0
        
        # accelerator
        self.accelerator = Accelerator(
            split_batches = True,
            mixed_precision = 'no'
        )

        # EMA model
        if self.accelerator.is_main_process:
            self.ema = EMA(imle_generator, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        # optimizer
        self.opt = build_optimizer(self.imle_model, self.cfg.OPTIMIZATION)
        self.scheduler = build_scheduler(self.opt, self.cfg.OPTIMIZATION, len(self.train_loader))

        # prepare model, dataloader, optimizer with accelerator
        self.imle_model, self.opt = self.accelerator.prepare(self.imle_model, self.opt)

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
        self.print_model_params(self.imle_model, 'Stage Two Model')

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
            'model': self.accelerator.get_state_dict(self.imle_model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }
        torch.save(data, os.path.join(self.cfg.model_dir, f'{ckpt_name}.pt'))

    def save_last_ckpt(self):
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.imle_model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        torch.save(data, os.path.join(self.cfg.model_dir, 'checkpoint_last.pt'))
    
    def load(self, ckpt_name):
        accelerator = self.accelerator

        data = torch.load(os.path.join(self.cfg.model_dir, f'{ckpt_name}.pt'), map_location=self.device, weights_only=True)

        model = self.accelerator.unwrap_model(self.imle_model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            # pass
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        """
        Training loop
        """

        # init
        accelerator = self.accelerator
        self.logger.info('training start')
        iter_per_epoch = self.train_num_steps // self.cfg.OPTIMIZATION.NUM_EPOCHS

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                # init per-iteration variables
                total_loss = 0.
                self.imle_model.train()
                self.ema.ema_model.train()

                for _ in range(self.gradient_accumulate_every):
                    data = {k : v.to(self.device) for k, v in next(self.dl).items()}
                    
                    log_dict = {'cur_epoch': self.step // iter_per_epoch}

                    if self.cfg.get('perturb_ctx', 0.0):
                        # used in SDD dataset
                        bs = data['past_traj'].shape[0]
                        scale_ = torch.randn((bs), device=self.device) * self.cfg.perturb_ctx + 1
                        data['past_traj_original_scale'] = data['past_traj_original_scale'] * scale_[:, None, None, None]

                    # compute the loss
                    with self.accelerator.autocast():
                        loss, loss_chamfer, loss_gt = self.imle_model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                    # log to tensorboard
                    if self.tb_log is not None:
                        self.tb_log.add_scalar('train/loss_total', loss.item(), self.step)
                        self.tb_log.add_scalar('train/loss_chamfer', loss_chamfer.item(), self.step)
                        self.tb_log.add_scalar('train/loss_gt', loss_gt.item(), self.step)
                        self.tb_log.add_scalar('train/learning_rate', self.opt.param_groups[0]["lr"], self.step)
                    
                pbar.set_description(f'total loss: {total_loss:.4f}, chamfer loss: {loss_chamfer.item():.4f}, gt loss: {loss_gt.item():.4f}, lr: {self.opt.param_groups[0]["lr"]:.6f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.imle_model.parameters(), self.cfg.OPTIMIZATION.GRAD_NORM_CLIP)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

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
        distances: [b*num_agents, k_preds, future_frames] or [b*num_agents, timestamps, k_preds, future_frames]
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
        predictions = pred_trajs[:, :, :end_frame, :].permute(0, 2, 1, 3)  # Shape: (B, T, N, D)

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

        if mode == 'last':
            ckpt_states = torch.load(os.path.join(self.cfg.model_dir, 'checkpoint_last.pt'), map_location=self.device, weights_only=True)
        else:
            ckpt_states = torch.load(os.path.join(self.cfg.model_dir, 'checkpoint_best.pt'), map_location=self.device, weights_only=True)

        self.imle_model = self.accelerator.unwrap_model(self.imle_model)
        self.imle_model.load_state_dict(ckpt_states['model'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(ckpt_states["ema"])
  
        # testing_mode=False, training_err_check=False
        if eval_on_train:
            fut_traj_gt, _, _ = self.eval_dataloader(training_err_check=True)
        else:
            fut_traj_gt, _, _ = self.eval_dataloader(testing_mode=True)
        self.logger.info(f'testing complete with the {mode} ckpt')


    def sample_from_imle(self, data):
        """
        Return the samples from denoising model in normal scale
        """

        pred_traj = self.imle_model(data, num_to_gen=1)
        pred_traj = pred_traj.squeeze(1)

        if self.cfg.dataset == 'nba':
            assert list(pred_traj.shape[2:]) == [self.cfg.agents, 40]
        elif self.cfg.dataset in ['eth_ucy', 'sdd']:
            assert list(pred_traj.shape[2:]) == [self.cfg.agents, 24]

        pred_traj = rearrange(pred_traj, 'b k a (f d) -> (b a) k f d', f=self.cfg.future_frames)[...,0:2]  # [B, k_preds, 11, 40] -> [B * 11, k_preds, 20, 2]

        if self.cfg.get('data_norm', None) == 'min_max':
            pred_traj = unnormalize_min_max(pred_traj, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1) 
        elif self.cfg.get('data_norm', None) == 'sqrt':
            pred_traj = unnormalize_sqrt(pred_traj, self.sqrt_a_, self.sqrt_b_)

        return pred_traj
    

    def save_latent_states(self, y_pred_data_ls, x_data_ls, file_name):
        self.logger.info("Begin to save the denoising samples...")

        if self.cfg.dataset in ['nba', 'sdd', 'eth_ucy']:
            keys_to_save = ['past_traj', 'fut_traj', 'past_traj_original_scale', 'fut_traj_original_scale', 'fut_traj_vel']
        else:
            raise NotImplementedError(f'Dataset [{self.cfg.dataset}] is not implemented yet.')
    
        states_to_save = {k: [] for k in keys_to_save}

        states_to_save['y_pred_data'] = []

        for i_batch, (y_pred_data, x_data) in enumerate(zip(y_pred_data_ls, x_data_ls)):
            y_pred_data = y_pred_data.detach().cpu().numpy()
            states_to_save['y_pred_data'].append(y_pred_data)

            for key in keys_to_save:
                x_data_val_ = x_data[key].detach().cpu().numpy()
                assert len(y_pred_data) == len(x_data_val_)
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

        num_datapoints = len(states_to_save['y_pred_data'])
        meta_data = {'cfg': cfg_, 'size': num_datapoints}

        states_to_save['meta_data'] = meta_data
        
        # save_path = os.path.join(self.cfg.sample_dir, f'{file_name}.npz')
        # np.savez_compressed(save_path, **states_to_save)

        save_path = os.path.join(self.cfg.sample_dir, f'{file_name}.pkl')
        self.logger.info("Saving the IMLE samples to {}".format(save_path))
        pickle.dump(states_to_save, open(save_path, 'wb'))


    def eval_dataloader(self, testing_mode=False, training_err_check=False):
        """
        General API to evaluate the dataloader/dataset
        """
        ### turn on the eval mode
        self.imle_model.eval()   
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
        performance = {'FDE_min': [0,0,0,0], 'ADE_min': [0,0,0,0], 'FDE_avg': [0,0,0,0], 'ADE_avg': [0,0,0,0], 'A_var': [0,0,0,0], 'F_var': [0,0,0,0], 'MASD': [0,0,0,0]}
        performance_joint = {'JFDE_min': [0,0,0,0], 'JADE_min': [0,0,0,0], 'JFDE_avg': [0,0,0,0], 'JADE_avg': [0,0,0,0]}
        num_trajs = 0
        ### record running time 
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i_batch, data in enumerate(dl): 
            bs = int(data['batch_size'])
            data = {k : v.to(self.device) for k, v in data.items()}

            pred_traj = self.sample_from_imle(data)

            fut_traj = rearrange(data['fut_traj_original_scale'], 'b a f d -> (b a) f d')               # [B, A, T, F] -> [B * A, T, F]
            fut_traj_gt = fut_traj.unsqueeze(1).repeat(1, self.cfg.denoising_head_preds, 1, 1)          # [B * A, K, T, F]
            distances = (fut_traj_gt - pred_traj).norm(p=2, dim=-1)                                     # [B * A, K, T]


            if self.cfg.dataset == 'nba':
                freq = 5 
                factor_time = 1
            elif self.cfg.dataset == 'eth_ucy':
                freq = 3
                factor_time = 1.2
            elif self.cfg.dataset == 'sdd':
                freq = 3
                factor_time = 1.2
                
            for time in range(1, 5):
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
             
            num_trajs += fut_traj.shape[0]

            # save the imle samples
            if self.save_samples:
                pred_traj = rearrange(pred_traj, '(b a) k f d -> b k a f d', b=bs)  # [B, K, A, T, F]
            
                num_datapoints = len(pred_traj)

                y_pred_data_ls = [pred_traj]
                x_data_ls = [data]

                solver_tag = self.cfg.get('solver_tag', '')
                save_name = f'imle_samples_{status}_batch_{i_batch}_{num_datapoints}_{solver_tag}'
                self.save_latent_states(y_pred_data_ls, x_data_ls, save_name)
                
                y_pred_data_ls, x_data_ls = [], []
        end.record()
        torch.cuda.synchronize()
        self.logger.info(f'Time elapsed: {start.elapsed_time(end):.5f} ms')
        self.logger.info(f'Time elapsed per scene: {start.elapsed_time(end)/len(dl.dataset):.5f} ms')
        self.logger.info(f'Number of scenes: {len(dl.dataset)}')
        cur_epoch = self.step // (self.train_num_steps // self.cfg.OPTIMIZATION.NUM_EPOCHS)
        if not testing_mode: 
            self.logger.info(f'{self.step}/{self.train_num_steps}, running inference on {num_trajs} agents (trajectories)')
            for time in range(4):
                if self.tb_log:
                    self.tb_log.add_scalar(f'eval_{status}/ADE_min_{(time+1)*factor_time:.1f}s', performance['ADE_min'][time]/num_trajs, cur_epoch)
                    self.tb_log.add_scalar(f'eval_{status}/FDE_min_{(time+1)*factor_time:.1f}s', performance['FDE_min'][time]/num_trajs, cur_epoch)
                    self.tb_log.add_scalar(f'eval_{status}/ADE_avg_{(time+1)*factor_time:.1f}s', performance['ADE_avg'][time]/num_trajs, cur_epoch)
                    self.tb_log.add_scalar(f'eval_{status}/FDE_avg_{(time+1)*factor_time:.1f}s', performance['FDE_avg'][time]/num_trajs, cur_epoch)
                    self.tb_log.add_scalar(f'eval_{status}/JADE_min_{(time+1)*factor_time:.1f}s', performance_joint['JADE_min'][time]/num_trajs, cur_epoch)
                    self.tb_log.add_scalar(f'eval_{status}/JFDE_min_{(time+1)*factor_time:.1f}s', performance_joint['JFDE_min'][time]/num_trajs, cur_epoch)

        # print out the performance
        for time in range(4):
            self.logger.info('--ADE_min({:.1f}s): {:.7f}\t--FDE_min({:.1f}s): {:.7f}'.format(
                time+1, performance['ADE_min'][time]/num_trajs, (time+1)*factor_time, performance['FDE_min'][time]/num_trajs))

      
        for time in range(4):
            self.logger.info('--ADE_avg({:.1f}s): {:.7f}\t--FDE_avg({:.1f}s): {:.7f}'.format(
                time+1, performance['ADE_avg'][time]/num_trajs, (time+1)*factor_time, performance['FDE_avg'][time]/num_trajs))
        
        for time in range(4):
            self.logger.info('--AVar({:.1f}s): {:.7f}\t--FVar({:.1f}s): {:.7f}'.format(
                time+1, performance['A_var'][time]/num_trajs, time+1, performance['F_var'][time]/num_trajs))

        for time in range(4):
            self.logger.info('--MASD({:.1f}s): {:.7f}'.format(
                time+1, performance['MASD'][time]/num_trajs))

        # print out the joint performance
        for time in range(4):
            self.logger.info('--JADE_min({:.1f}s): {:.7f}\t--JFDE_min({:.1f}s): {:.7f}'.format(
                time+1, performance_joint['JADE_min'][time]/num_trajs, (time+1)*factor_time, performance_joint['JFDE_min'][time]/num_trajs))
            
        for time in range(4):
            self.logger.info('--JADE_avg({:.1f}s): {:.7f}\t--JFDE_avg({:.1f}s): {:.7f}'.format(
                time+1, performance_joint['JADE_avg'][time]/num_trajs, (time+1)*factor_time, performance_joint['JFDE_avg'][time]/num_trajs))

        return fut_traj_gt, performance, num_trajs

