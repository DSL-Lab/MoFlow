import numpy as np

import torch
import torch.nn as nn
from .context_encoder import build_context_encoder
from .motion_decoder import build_decoder
from .motion_decoder.mtr_decoder import modulate
from .utils.common_layers import build_mlps
from einops import repeat, rearrange
from models.context_encoder.mtr_encoder import SinusoidalPosEmb
from models.context_encoder.condition_encoder import ZEncoder, ZFiLM
from models.neural_sde.ctr_sde import simulate_sde_paths, ControlledSSLSDE
from models.neural_sde.z0_encoder import Z0Encoder


class MotionTransformer(nn.Module):
    def __init__(self, model_config, logger, config):
        super().__init__()
        self.model_cfg = model_config
        self.dim = self.model_cfg.CONTEXT_ENCODER.D_MODEL # 统一的通道维（上下游对齐）
        self.config = config
        self.logger = logger
        use_pre_norm = self.model_cfg.get('USE_PRE_NORM', False)

        assert not use_pre_norm, "Pre-norm is not supported in this model"
        self.T_f = self.config.get('future_frames', 0)
        self.dt = self.config.get('dt', 0)
        # （1）上下文编码器：把历史轨迹/邻居信息编码成每个 agent 的上下文向量
        self.context_encoder = build_context_encoder(self.model_cfg.CONTEXT_ENCODER, use_pre_norm, config.device)

        # update 2: 将原有的条件/指令编码器 + 注入层 改为神经动力系统
        # if code_version == "1.0":
            # self.z_encoder = ZEncoder(
            #     d_hist = self.model_cfg.get('COND_D_HIST', 0),
            #     d_cue = self.model_cfg.get('COND_D_CUE', 0),
            #     d_model = self.dim,
            #     d_z = self.model_cfg.get('COG_D_Z', 0)
            # )

            # 最小侵入：一对 FiLM 调制 + 一个线性投影用于拼接
            # self.cond_proj       = nn.Linear(self.dim, self.dim)
            # self.z_proj = nn.Linear(self.model_cfg.get('COG_D_Z', 0), self.dim)
            # self.z_film = ZFiLM(d_feat=self.dim)
            # self.z_gamma = nn.Linear(self.dim, self.dim)
            # self.z_beta = nn.Linear(self.dim, self.dim)

            # nn.init.zeros_(self.z_gamma.weight)
            # nn.init.zeros_(self.z_gamma.bias)
            # nn.init.zeros_(self.z_beta.weight)
            # nn.init.zeros_(self.z_beta.bias)

            # self.cond_film_gamma = nn.Linear(self.dim, self.dim)
            # self.cond_film_beta  = nn.Linear(self.dim, self.dim)

        self.z0_encoder = Z0Encoder(
            num_keypoints=self.config.agents,
            kp_dim=9,
            stim_dim=self.model_cfg.get('COND_D_CUE', 0),
            hidden_dim=self.dim,
            z_dim=self.model_cfg.get('COG_D_Z', 0)
        )

        self.neural_sde = ControlledSSLSDE(
            z_dim=self.model_cfg.get('COG_D_Z', 0),
            stim_dim=self.model_cfg.get('COND_D_CUE', 0),
            num_regimes=3,
            num_bases=16,
            hidden_dim=self.dim,
            init_scale=0.1,
        )
        self.z_seq_proj =  nn.Linear(self.model_cfg.get('COG_D_Z', 0), self.dim)
        self.z_seq_gamma = nn.Linear(self.dim, self.dim)
        self.z_seq_beta = nn.Linear(self.dim, self.dim)

        # （2）“位置编码”的三件套：K 维query索引、A维agent索引、以及一个事后融合MLP
        self.motion_query_embedding = nn.Embedding(self.model_cfg.NUM_PROPOSED_QUERY, self.dim)
        self.agent_order_embedding = nn.Embedding(self.model_cfg.CONTEXT_ENCODER.NUM_OF_ATTN_NEIGHBORS, self.dim)
        self.post_pe_cat_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )
        # 说明：这里把 query/agent 的PE与 token 融合，作为 decoder 的输入 token

        # （3）时间嵌入：正弦时间编码 + 两层MLP，输出维度设为 time_dim(这些奇奇怪怪的编码都是什么?)
        time_dim = self.dim * 1
        sinu_pos_emb = SinusoidalPosEmb(self.dim, theta = 10000)

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(self.dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim),
        )

        # （4）噪声/中间态 y 的嵌入：把 [T*D] 维的向量映到 denoiser 的通道维
        self.noisy_y_mlp = nn.Sequential(
            nn.Linear(self.model_cfg.MODEL_OUT_DIM, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )

        # （5）两路自注意：沿 K（不同proposal分支）与沿 A（不同agent）做 self-attn
        dropout_ = self.model_cfg.MOTION_DECODER.DROPOUT_OF_ATTN
        self.noisy_y_attn_k = nn.TransformerEncoderLayer(d_model=self.dim, nhead=4, dim_feedforward=self.dim * 4, dropout=dropout_, batch_first=True)
        self.noisy_y_attn_a = nn.TransformerEncoderLayer(d_model=self.dim, nhead=4, dim_feedforward=self.dim * 4, dropout=dropout_, batch_first=True)

        # （6）把三路信息拼接后（context、y_emb、t_emb）先做一个融合，再送 decoder
        dim_decoder = self.model_cfg.MOTION_DECODER.D_MODEL
        # ⚠️ 因为要把 cond 拼到融合前，所以把 in_features + self.dim
        in_fuse = (self.dim            # encoder_out
                   + (self.dim*1)      # time_dim
                   + self.dim)          # y_emb
                   # + self.dim)         # cond_vec
        self.init_emb_fusion_mlp = nn.Sequential(
            nn.Linear(in_fuse, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, dim_decoder),
        )

        # （7）decoder 的读出层：回归向量（速度/残差/去噪方向）与分类分支（best-of-K用）
        self.readout_mlp = nn.Sequential(
            nn.Linear(dim_decoder, dim_decoder),
            nn.ReLU(),
            nn.Linear(dim_decoder, self.model_cfg.MODEL_OUT_DIM),
        )

        self.motion_decoder = build_decoder(self.model_cfg.MOTION_DECODER, use_pre_norm)

        self.reg_head = build_mlps(c_in=self.dim, mlp_channels=self.model_cfg.REGRESSION_MLPS, ret_before_act=True, without_norm=True)
        self.cls_head = build_mlps(c_in=dim_decoder, mlp_channels=self.model_cfg.CLASSIFICATION_MLPS, ret_before_act=True, without_norm=True)
        # 说明：你这里有两套 readout：一个是 reg_head（输入 dim，输出 T*D），
        # 另一个是 cls_head（输入 decoder维，输出 K类logit）。实际 forward 里 reg_head 用在 decoder 之后。
        # （8）统计参数量，便于日志观测
        params_encoder = sum(p.numel() for p in self.context_encoder.parameters())
        params_decoder = sum(p.numel() for p in self.motion_decoder.parameters())
        params_total = sum(p.numel() for p in self.parameters())
        params_other = params_total - params_encoder - params_decoder
        logger.info("Total parameters: {:,}, Encoder: {:,}, Decoder: {:,}, Other: {:,}".format(
            params_total, params_encoder, params_decoder, params_other
        ))

    # -------- 构造 SDE 的未来控制序列 u_seq --------
    def _build_future_control_seq(
            self,
            x_data,
            B: int,
            device: torch.device,
            dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        构造用于 SDE 的未来控制序列 u_seq: [B, T_f, stim_dim]

        推荐做法：
            如果 x_data 里已经有未来刺激计划 (如 x_data['fut_stim'])，
            则直接使用；
            否则简单使用最后一个历史刺激帧重复未来 T_f 次，作为
            近似控制输入（也是一个合理的 baseline）。
        """
        T_f = self.T_f

        # 情况 1：有明确的未来刺激序列
        if "fut_cond_cue" in x_data:
            # print("--- found fut_cond_cue")
            fut_stim = x_data["fut_cond_cue"]  # [B, T_f, stim_dim]
            # print("fut_cond_cue shape = {} {}".format(fut_stim.shape, T_f))
            assert fut_stim.shape[0] == B
            assert fut_stim.shape[1] == T_f
            assert fut_stim.shape[2] == self.neural_sde.stim_dim
            u_seq = fut_stim.to(device=device, dtype=dtype)
        else:
            # 情况 2：没有未来刺激，则用最后一个历史刺激值重复
            assert "hist_stim" in x_data, "需要在 x_data 中提供 hist_stim 或 fut_stim"
            hist_stim = x_data["hist_stim"]  # [B, Th, stim_dim]
            u_last = hist_stim[:, -1, :]  # [B, stim_dim]
            u_seq = u_last.unsqueeze(1).repeat(1, T_f, 1)  # [B, T_f, stim_dim]

        return u_seq

    def apply_PE(self, y_emb, k_pe_batch, a_pe_batch):
        '''
        Apply positional encoding to the input embeddings according to self.model_cfg. This is used for ablation study.
        根据开关把 K（proposal）PE、A（agent）PE 加到 token 上。
        做消融时可以只开其中之一或全关。
        '''
        # print("y_emb shape = {}, k_pe_batch shape = {}, a_pe_batch shape = {}".format(
        #     y_emb.shape, k_pe_batch.shape, a_pe_batch.shape
        # ))
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
        y: 噪声或中间态（如 x_t 或 y_t），形状 [B, K, A, T*D]
        x_data: 数据字典（包含历史轨迹、原尺度等）
        time: 连续时间步（CFM/F.M. 的 t）
        '''
        ### Rat assertions
        # assert list(y.shape[2:]) == [8, 30, 2] or list(y.shape[2:]) == [8, 60], 'y shape is not correct'
        # if y.size(-1) == 2:
        #     y = y.reshape((-1, 20, 8, 40))

        device = y.device
        # print("y device = {}".format(device))
        B, K, A, _ = y.shape

        # （1）上下文编码：根据历史（及邻居）得到每个 agent 的上下文向量 [B, A, D]
        encoder_out = self.context_encoder(x_data['past_traj_original_scale'])  # [B, A, D]
        # 扩到 [B, K, A, D] 与 proposal 维对齐
        encoder_out_batch = repeat(encoder_out, 'b a d -> b k a d', k=K, a=A) 	# [B, K, A, D]
        # encoder_out_batch = encoder_out_batch.unsqueeze(3).repeat(1, 1, 1, self.T_f, 1)
        # print("encoder_out_batch shape = {}".format(encoder_out_batch.shape))

        # update: 1.5 条件编码
        # if code_version == "1.0":
            # cond_flow, z = self.z_encoder(x_data)  # Stage A
            # cond_flow = self.cond_proj(cond_flow)
            # cond_bka = repeat(cond_flow, 'b d -> b k a d', k=K, a=A)
            # z_feat = self.z_proj(z)
            # z_bka = z_feat[:, None, None, :].expand(B, K, A, -1)      # [B,K,A,d_model]

        past_traj = x_data["past_traj"]
        hist_stim = x_data["hist_cond_cue"]
        # 2.1) 编码初始隐变量 z0
        z0 = self.z0_encoder(past_traj, hist_stim)   # [B, z_dim]

        # 2.2) 构造未来控制序列 u_seq 并仿真 SDE 得到 z_seq
        u_seq = self._build_future_control_seq(
            x_data=x_data,
            B=B,
            device=device,
            dtype=z0.dtype,
        )                                          # [B, T_f, stim_dim]

        z_seq = simulate_sde_paths(
            sde=self.neural_sde,
            z0=z0,
            u_seq=u_seq,
            dt=self.dt,
        )                                          # [B, T_f, d_dim]
        z_frame = self.z_seq_proj(z_seq)
        z_frame_bka = z_frame[:, None, None, :, :].expand(B, K, A, self.T_f, self.dim)

        gamma = self.z_seq_gamma(z_frame_bka)  # [B,K,A,T_f,D]
        beta = self.z_seq_beta(z_frame_bka)  # [B,K,A,T_f,D]

        # 到这里位置没有问题，下面如何进行特征融合？

        # （2）把 y（T*D）嵌入到通道维：得到每个 (B,K,A) 的 token
        y_emb = self.noisy_y_mlp(y)  	# [B, K, A, D]
        # y_emb = y_emb.unsqueeze(3).repeat(1, 1, 1, self.T_f, 1)

        # if code_version == "1.0":
            # gamma, beta = self.cond_film_gamma(cond_bka), self.cond_film_beta(cond_bka)
            # y_emb = gamma * y_emb + beta

        # （3）时间嵌入：若方法是 fm，则把 t 放大到更大的数值（经验 trick），再过 time_mlp
        time_ = time
        if self.config.denoising_method == 'fm':
            time = time * 1000.0  # flow matching time upscaling

        t_emb = self.time_mlp(time) 	# [B, D]
        t_emb_batch = repeat(t_emb, 'b d -> b k a d', b=B, k=K, a=A) # [B, K, A, D]  # 与 (K,A) 对齐
        # t_emb_batch = t_emb_batch.unsqueeze(3).repeat(1, 1, 1, self.T_f, 1)

        # （4）构造 K、A 的“索引型位置编码”并扩展到 batch
        k_pe = self.motion_query_embedding(torch.arange(self.model_cfg.NUM_PROPOSED_QUERY, device=device))	# [K, D]
        k_pe_batch = repeat(k_pe, 'k d -> b k a d', b=B, a=A)	# [B, K, A, D]
        # k_pe_batch = k_pe_batch.unsqueeze(3).repeat(1, 1, 1, self.T_f, 1)

        a_pe = self.agent_order_embedding(torch.arange(self.model_cfg.CONTEXT_ENCODER.NUM_OF_ATTN_NEIGHBORS, device=device))  # [A, D]
        a_pe_batch = repeat(a_pe, 'a d -> b k a d', b=B, k=K)	# [B, K, A, D]
        # a_pe_batch = a_pe_batch.unsqueeze(3).repeat(1, 1, 1, self.T_f, 1)

        # （5）对 y_emb 分别沿 K、沿 A 做自注意，增强 proposal间/agent间的信息交互
        # 先加PE再按 K 维重排为序列：(b a) 为批次，长度 K
        y_emb_k = rearrange(self.apply_PE(y_emb, k_pe_batch, a_pe_batch), 'b k a d -> (b a) k d')
        # print("y_emb_k shape = {}".format(y_emb_k.shape))
        y_emb_k = self.noisy_y_attn_k(y_emb_k)
        y_emb = rearrange(y_emb_k, '(b a) k d -> b k a d', b=B, a=A)

        # 再按 A 维重排为序列：(b k) 为批次，长度 A
        y_emb_a = rearrange(y_emb, 'b k a d -> (b k) a d')
        y_emb_a = self.noisy_y_attn_a(y_emb_a)
        y_emb = rearrange(y_emb_a, '(b k) a d -> b k a d', b=B, k=K)

        # （6）训练时可选的 embedding 级丢弃（按 t 的逻辑概率掩蔽整个 token）
        if self.training and self.config.get('drop_method', None) == 'emb':
            assert self.config.get('drop_logi_k', None) is not None and self.config.get('drop_logi_m', None) is not None
            m, k = self.config.drop_logi_m, self.config.drop_logi_k
            p_m = 1 / (1 + torch.exp(-k * (time_ - m)))
            p_m = p_m[:, None, None, None]
            y_emb = y_emb.masked_fill(torch.rand_like(p_m) < p_m, 0.)


        # （7）与上下文、时间一起融合；再加一次 PE 后交给 motion decoder
        # update
        # emb_in = torch.cat((encoder_out_batch, y_emb, t_emb_batch, cond_bka), dim=-1)
        emb_in = torch.cat((encoder_out_batch, y_emb, t_emb_batch), dim=-1)
        emb_fusion = self.init_emb_fusion_mlp(emb_in)	 	# [B, K, A, D]
        # # 11-24 尝试在此处融合，效果显著，出于对比考虑修改
        emb_fusion = emb_fusion.unsqueeze(3).repeat(1, 1, 1, self.T_f, 1) # [B, K, A, T, D]
        emb_fusion = emb_fusion * (1 + gamma) + beta        # [B, K, A, T, D]
        a_pe_batch = a_pe_batch.unsqueeze(3).repeat(1, 1, 1, self.T_f, 1)
        k_pe_batch = k_pe_batch.unsqueeze(3).repeat(1, 1, 1, self.T_f, 1)

        query_token = self.post_pe_cat_mlp(self.apply_PE(emb_fusion, k_pe_batch, a_pe_batch)) 								# [B, K, A, D]
        # print("query token shape = {}".format(query_token.shape))
        # query_token = rearrange(query_token, 'b k a t d -> b (k a t) d')
        readout_token = self.motion_decoder(query_token, t_emb)													# [B, K, A, D]
        # print("readout token shape = {}".format(readout_token.shape))

        # # 11-25 尝试在后期融合z
        # readout_token = readout_token.unsqueeze(3).repeat(1, 1, 1, self.T_f, 1) # [B, K, A, T, D]
        # readout_token = readout_token * (1.0 + gamma) + beta  # [B,K,A,T_f,D]

        # （8）读出：回归分支输出 T*D，分类分支输出 [B,K,A] 的打分
        denoiser_x = self.reg_head(readout_token)  										# [B, K, A, F * D]
        denoiser_x = rearrange(denoiser_x, 'b k a t d -> b k a (t d)')
        # print("denoiser_x shape = {}".format(denoiser_x.shape))

        # denoiser_cls = self.cls_head(readout_token).squeeze(-1) 						# [B, K, A]

        return denoiser_x


class IMLETransformer(nn.Module):
    def __init__(self, model_config, logger, config):
        super().__init__()
        self.model_cfg = model_config
        self.dim = self.model_cfg.CONTEXT_ENCODER.D_MODEL
        self.cfg = config

        self.objective = self.cfg.objective

        use_pre_norm = self.model_cfg.get('USE_PRE_NORM', False)

        assert not use_pre_norm, "Pre-norm is not supported in this model"

        self.context_encoder = build_context_encoder(self.model_cfg.CONTEXT_ENCODER, use_pre_norm, config.device)

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
