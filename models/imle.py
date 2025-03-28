import os
import numpy as np
import matplotlib.pyplot as plt

from einops import rearrange

import torch
import torch.nn.functional as F
from torch import nn
from utils.normalization import unnormalize_min_max, unnormalize_sqrt


class IMLE(nn.Module):
    def __init__(self, cfg, model, logger):        
        super(IMLE, self).__init__()
        self.cfg = cfg
        self.model = model
        self.logger = logger

    def forward(self, x_data, num_to_gen=1):
        """
        Train the IMLE generator.
        """

        # Get the model's predictions
        imle_gen = self.model(x_data, num_to_gen)  # [B, M, K, A, F * D]

        imle_gen_metric = unnormalize_min_max(imle_gen, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1)

        if self.cfg.objective == 'set':
            pass
        else:
            raise NotImplementedError(f"Objective {self.cfg.objective} not implemented.")

        if self.training:
            # Compute the loss
            target = x_data['y_t']              # [B, K, A, T, 2]
            target_metric = unnormalize_min_max(target, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1)

            B, K, A, T, _ = target_metric.shape
            M = imle_gen_metric.shape[1]


            if self.cfg.objective == 'set':
                # compute chamfer distance between target and generated trajectories

                imle_gen_metric = imle_gen_metric.view(B, M, K, A, T, 2)

                if self.cfg.get('loss_reg_chamfer_weight', 0.0):
                    # compute chamfer distance between two sets of trajectories
                    imle_gen_metric_ = rearrange(imle_gen_metric, 'B M K A T D -> B A M K 1 T D')  # [B, A, M, K1, 1, T, 2]
                    target_metric_ = rearrange(target_metric, 'B K A T D -> B A 1 1 K T D')        # [B, A, 1, 1, K2, T, 2]

                    pairwise_dist = torch.norm(imle_gen_metric_ - target_metric_, dim=-1)          # [B, A, M, K1, K2, T]

                    if self.cfg.loss_reg_reduction == 'sum':
                        pairwise_dist = pairwise_dist.sum(dim=-1)   # [B, A, M, K1, K2]
                    elif self.cfg.loss_reg_reduction == 'mean':
                        pairwise_dist = pairwise_dist.mean(dim=-1)  # [B, A, M, K1, K2]

                    min_dist_imle_to_target_ = pairwise_dist.min(dim=-1)[0]  # [B, A, M, K1], Minimum along target for each imle point
                    min_dist_target_to_imle_ = pairwise_dist.min(dim=-2)[0]  # [B, A, M, K2], Minimum along imle for each target point

                    # mean distance over trajectories
                    chamfer_dist = min_dist_imle_to_target_.mean(dim=-1) + min_dist_target_to_imle_.mean(dim=-1)  # [B, A, M]

                    # mean distance over agents
                    chamfer_dist_m = chamfer_dist.mean(dim=1)  # [B, M]

                    loss_chamfer = chamfer_dist_m.min(dim=-1)[0].mean() * self.cfg.loss_reg_chamfer_weight
                else:
                    loss_chamfer = torch.tensor(0.0).to(imle_gen.device)

                # gt supervision
                if self.cfg.get('loss_reg_gt_weight', 0.0):
                    gt_metric = x_data['fut_traj_original_scale']  # [B, A, T, 2]
                    gt_metric_ = rearrange(gt_metric, 'B A T D -> B 1 1 A T D')

                    imle_gen_to_gt_dist = torch.norm(imle_gen_metric - gt_metric_, dim=-1)  # [B, M, K, A, T]

                    if self.cfg.loss_reg_reduction == 'sum':
                        imle_gen_to_gt_dist = imle_gen_to_gt_dist.sum(dim=-1)       # [B, M, K, A]
                    elif self.cfg.loss_reg_reduction == 'mean':
                        imle_gen_to_gt_dist = imle_gen_to_gt_dist.mean(dim=-1)      # [B, M, K, A]

                    imle_gen_nn_gt = imle_gen_to_gt_dist.min(dim=-2)[0].mean(dim=-1)    # [B, M]

                    # gt_chosen_m = chamfer_dist_m.argmin(dim=-1)
                    gt_chosen_m = imle_gen_nn_gt.argmin(dim=-1)

                    loss_gt = imle_gen_nn_gt[torch.arange(B), gt_chosen_m].mean() * self.cfg.loss_reg_gt_weight
                else:
                    loss_gt = torch.tensor(0.0).to(imle_gen.device)

                loss = loss_chamfer + loss_gt
        
            return loss, loss_chamfer, loss_gt
        else:
            return imle_gen