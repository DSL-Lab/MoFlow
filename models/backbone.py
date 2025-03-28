import numpy as np

import torch
import torch.nn as nn
from .context_encoder import build_context_encoder
from .motion_decoder import build_decoder
from .motion_decoder.mtr_decoder import modulate
from .utils.common_layers import build_mlps
from einops import repeat, rearrange
from models.context_encoder.mtr_encoder import SinusoidalPosEmb


class MotionTransformer(nn.Module):
    def __init__(self, model_config, logger, config):
        super().__init__()
        self.model_cfg = model_config
        self.dim = self.model_cfg.CONTEXT_ENCODER.D_MODEL
        self.config = config

        use_pre_norm = self.model_cfg.get('USE_PRE_NORM', False)

        assert not use_pre_norm, "Pre-norm is not supported in this model"

        self.context_encoder = build_context_encoder(self.model_cfg.CONTEXT_ENCODER, use_pre_norm)

        ### serves the purpose of positional encoding
        self.motion_query_embedding = nn.Embedding(self.model_cfg.NUM_PROPOSED_QUERY, self.dim)
        self.agent_order_embedding = nn.Embedding(self.model_cfg.CONTEXT_ENCODER.NUM_OF_ATTN_NEIGHBORS, self.dim)
        self.post_pe_cat_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )

        time_dim = self.dim * 1
        sinu_pos_emb = SinusoidalPosEmb(self.dim, theta = 10000)

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(self.dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.noisy_y_mlp = nn.Sequential(
            nn.Linear(self.model_cfg.MODEL_OUT_DIM, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )

        dropout_ = self.model_cfg.MOTION_DECODER.DROPOUT_OF_ATTN
        self.noisy_y_attn_k = nn.TransformerEncoderLayer(d_model=self.dim, nhead=4, dim_feedforward=self.dim * 4, dropout=dropout_, batch_first=True)
        self.noisy_y_attn_a = nn.TransformerEncoderLayer(d_model=self.dim, nhead=4, dim_feedforward=self.dim * 4, dropout=dropout_, batch_first=True)

        dim_decoder = self.model_cfg.MOTION_DECODER.D_MODEL
        self.init_emb_fusion_mlp = nn.Sequential(
            nn.Linear(self.dim + time_dim + self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, dim_decoder),
        )
        
        self.readout_mlp = nn.Sequential(
            nn.Linear(dim_decoder, dim_decoder),
            nn.ReLU(),
            nn.Linear(dim_decoder, self.model_cfg.MODEL_OUT_DIM),
        )

        self.motion_decoder = build_decoder(self.model_cfg.MOTION_DECODER, use_pre_norm)

        self.reg_head = build_mlps(c_in=self.dim, mlp_channels=self.model_cfg.REGRESSION_MLPS, ret_before_act=True, without_norm=True)
        self.cls_head = build_mlps(c_in=dim_decoder, mlp_channels=self.model_cfg.CLASSIFICATION_MLPS, ret_before_act=True, without_norm=True)

        # print out the number of parameters
        params_encoder = sum(p.numel() for p in self.context_encoder.parameters())
        params_decoder = sum(p.numel() for p in self.motion_decoder.parameters())
        params_total = sum(p.numel() for p in self.parameters())
        params_other = params_total - params_encoder - params_decoder
        logger.info("Total parameters: {:,}, Encoder: {:,}, Decoder: {:,}, Other: {:,}".format(params_total, params_encoder, params_decoder, params_other))

    def apply_PE(self, y_emb, k_pe_batch, a_pe_batch):
        '''
        Apply positional encoding to the input embeddings according to self.model_cfg. This is used for ablation study.
        '''
        if self.model_cfg.get('USE_PE_QUERY', True) and self.model_cfg.get('USE_PE_AGENT', True):
            y_emb = y_emb + k_pe_batch + a_pe_batch
        elif self.model_cfg.get('USE_PE_QUERY', True):
            y_emb = y_emb + k_pe_batch
        elif self.model_cfg.get('USE_PE_AGENT', True):
            y_emb = y_emb + a_pe_batch
        else:
            pass
        return y_emb
    
    def forward(self, y, time, x_data):
        '''
        y: noisy vector
        x_data: data dict containing the following keys:
            - past_traj: past trajectory
            - future_traj: future trajectory
            - future_traj_vel: future trajectory velocity
            - trajectory mask: [it may exist]
            - batch_size: batch size
            - indexes: exist when we aim to perform IMLE
        time: denoising time step
        '''
        ### NBA assertions
        assert list(y.shape[2:]) == [11, 20, 2] or list(y.shape[2:]) == [11, 40], 'y shape is not correct'
        if y.size(-1) == 2:
            y = y.reshape((-1, 20, 11, 40))
        device = y.device
        B, K, A, _ = y.shape

        ### context encoder
        encoder_out = self.context_encoder(x_data['past_traj_original_scale'])  # [B, A, D]
        encoder_out_batch = repeat(encoder_out, 'b a d -> b k a d', k=K, a=A) 	# [B, K, A, D]


        ### init embeddings

   

        y_emb = self.noisy_y_mlp(y)  	# [B, K, A, D]

        time_ = time
        if self.config.denoising_method == 'fm':
            time = time * 1000.0  # flow matching time upscaling

        t_emb = self.time_mlp(time) 	# [B, D]
        t_emb_batch = repeat(t_emb, 'b d -> b k a d', b=B, k=K, a=A) # [B, K, A, D]

        k_pe = self.motion_query_embedding(torch.arange(self.model_cfg.NUM_PROPOSED_QUERY, device=device))	# [K, D]
        k_pe_batch = repeat(k_pe, 'k d -> b k a d', b=B, a=A)	# [B, K, A, D]

        a_pe = self.agent_order_embedding(torch.arange(self.model_cfg.CONTEXT_ENCODER.NUM_OF_ATTN_NEIGHBORS, device=device))  # [A, D]
        a_pe_batch = repeat(a_pe, 'a d -> b k a d', b=B, k=K)	# [B, K, A, D]


        y_emb_k = rearrange(self.apply_PE(y_emb, k_pe_batch, a_pe_batch), 'b k a d -> (b a) k d')
        y_emb_k = self.noisy_y_attn_k(y_emb_k)
        y_emb = rearrange(y_emb_k, '(b a) k d -> b k a d', b=B, a=A)

        y_emb_a = rearrange(y_emb, 'b k a d -> (b k) a d')
        y_emb_a = self.noisy_y_attn_a(y_emb_a)
        y_emb = rearrange(y_emb_a, '(b k) a d -> b k a d', b=B, k=K)

        if self.training and self.config.get('drop_method', None) == 'emb':
            assert self.config.get('drop_logi_k', None) is not None and self.config.get('drop_logi_m', None) is not None
            m, k = self.config.drop_logi_m, self.config.drop_logi_k
            p_m = 1 / (1 + torch.exp(-k * (time_ - m)))
            p_m = p_m[:, None, None, None]	
            y_emb = y_emb.masked_fill(torch.rand_like(p_m) < p_m, 0.)


        ### send to motion decoder
        emb_fusion = self.init_emb_fusion_mlp(torch.cat((encoder_out_batch, y_emb, t_emb_batch), dim=-1))	 	# [B, K, A, D]
        query_token = self.post_pe_cat_mlp(self.apply_PE(emb_fusion, k_pe_batch, a_pe_batch)) 								# [B, K, A, D]
        readout_token = self.motion_decoder(query_token, t_emb)													# [B, K, A, D]	

        ### readout layers
        denoiser_x = self.reg_head(readout_token)  										# [B, K, A, F * D]
        denoiser_cls = self.cls_head(readout_token).squeeze(-1) 						# [B, K, A]

        return denoiser_x, denoiser_cls


class IMLETransformer(nn.Module):
    def __init__(self, model_config, logger, config):
        super().__init__()
        self.model_cfg = model_config
        self.dim = self.model_cfg.CONTEXT_ENCODER.D_MODEL
        self.cfg = config

        self.objective = self.cfg.objective

        use_pre_norm = self.model_cfg.get('USE_PRE_NORM', False)

        assert not use_pre_norm, "Pre-norm is not supported in this model"

        self.context_encoder = build_context_encoder(self.model_cfg.CONTEXT_ENCODER, use_pre_norm)

        ### serves the purpose of positional encoding
        if self.objective == 'set':
            self.motion_query_embedding = nn.Embedding(self.model_cfg.NUM_PROPOSED_QUERY, self.dim)

        self.agent_order_embedding = nn.Embedding(self.model_cfg.CONTEXT_ENCODER.NUM_OF_ATTN_NEIGHBORS, self.dim)
        
        self.noisy_vec_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim)
        )

        self.pe_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )

        dim_decoder = self.model_cfg.MOTION_DECODER.D_MODEL
        self.init_emb_fusion_mlp = nn.Sequential(
            nn.Linear(self.dim + self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, dim_decoder),
        )
        
        self.readout_mlp = nn.Sequential(
            nn.Linear(dim_decoder, dim_decoder),
            nn.ReLU(),
            nn.Linear(dim_decoder, self.model_cfg.MODEL_OUT_DIM),
        )

        self.motion_decoder = build_decoder(self.model_cfg.MOTION_DECODER, use_pre_norm, use_adaln=False)

        self.reg_head = build_mlps(c_in=self.dim, mlp_channels=self.model_cfg.REGRESSION_MLPS, ret_before_act=True, without_norm=True)

        # print out the number of parameters
        params_encoder = sum(p.numel() for p in self.context_encoder.parameters())
        params_decoder = sum(p.numel() for p in self.motion_decoder.parameters())
        params_total = sum(p.numel() for p in self.parameters())
        params_other = params_total - params_encoder - params_decoder
        logger.info("Total parameters: {:,}, Encoder: {:,}, Decoder: {:,}, Other: {:,}".format(params_total, params_encoder, params_decoder, params_other))


    def forward(self, x_data, num_to_gen=None):
        device = x_data['past_traj_original_scale'].device
        B, A, T, _ = x_data['past_traj_original_scale'].shape
        K = self.cfg.denoising_head_preds
        D = self.dim

        if self.training:
            M = self.cfg.num_to_gen
        else:
            M = num_to_gen

        # context encoder
        encoder_out = self.context_encoder(x_data['past_traj_original_scale'])  # [B, A, D]

        # init noise embeddings
        noise = torch.randn((B, M, D), device=device)       # [B, M, D]
        noise_emb = self.noisy_vec_mlp(noise)  	            # [B, M, D]


        if self.cfg.objective == 'set':
            encoder_out_batch = repeat(encoder_out, 'b a d -> b m k a d', m=M, k=K, a=A)    # [B, M, K, A, D]

            k_pe = self.motion_query_embedding(torch.arange(K, device=device))	            # [K, D]
            k_pe_batch = repeat(k_pe, 'k d -> b m k a d', b=B, m=M, a=A)	                # [B, M, K, A, D]

            a_pe = self.agent_order_embedding(torch.arange(A, device=device))               # [A, D]
            a_pe_batch = repeat(a_pe, 'a d -> b m k a d', b=B, m=M, k=K)	                # [B, M, K, A, D]

            noise_emb_batch = repeat(noise_emb, 'b m d -> b m k a d', k=K, a=A)	            # [B, M, K, A, D]
        elif self.cfg.objective == 'single':
            raise NotImplementedError
        else:
            raise NotImplementedError

        # send to motion decoder
        emb_fusion = self.init_emb_fusion_mlp(torch.cat((encoder_out_batch, noise_emb_batch), dim=-1))	 	# [B, M, K, A, D]
        query_token = self.pe_mlp(emb_fusion + k_pe_batch + a_pe_batch) 					                # [B, M, K, A, D]

        if self.cfg.objective == 'set':
            query_token = rearrange(query_token, 'b m k a d -> (b m) k a d')
            readout_token = self.motion_decoder(query_token)
            readout_token = rearrange(readout_token, '(b m) k a d -> b m k a d', m=M)
        elif self.cfg.objective == 'single':
            raise NotImplementedError
        else:
            raise NotImplementedError

        # readout layers
        denoiser_x = self.reg_head(readout_token)  													# [B, K, A, F * D]

        return denoiser_x
