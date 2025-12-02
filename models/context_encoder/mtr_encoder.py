import numpy as np
import torch
import torch.nn as nn


from models.utils import polyline_encoder
from einops import rearrange
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta 

    def forward(self, x):
        device = x.device
        # print("x device = {}".format(device))
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MTREncoder(nn.Module):
    def __init__(self, config, use_pre_norm, device):
        super().__init__()
        self.model_cfg = config
        dim = self.model_cfg.D_MODEL

        # build polyline encoders
        self.agent_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.model_cfg.NUM_INPUT_CONTEXT,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_AGENT,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_AGENT,
            out_channels=dim
        ).to(device)
        # print("device = {}".format(device))

        # Positional encoding
        self.pos_encoding = nn.Sequential(
                SinusoidalPosEmb(dim, theta = 10000),
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            ).to(device)
        self.team_one_query_embedding = nn.Embedding(1, dim)
        self.team_two_query_embedding = nn.Embedding(1, dim)
        self.ball_query_embedding = nn.Embedding(1, dim)
        self.mlp_pe = nn.Sequential(
            nn.Linear(2*dim, dim),
            # nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        # build transformer encoder layers
        self.layer = nn.TransformerEncoderLayer(d_model=dim, 
                                                dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
                                                nhead=self.model_cfg.NUM_ATTN_HEAD, 
                                                dim_feedforward=dim * 4, 
                                                norm_first=use_pre_norm,
                                                batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=self.model_cfg.NUM_ATTN_LAYERS)
        self.num_out_channels = dim
        if self.model_cfg.DATA_TYPE == 'rat':
            self.max_agents = 8
        elif self.model_cfg.DATA_TYPE == 'babel':
            self.max_agents = 22
        self.agent_index_embed = nn.Embedding(self.max_agents, dim)

        # 定义关键点类型数，例如：0=后爪, 1=前爪, 2=头颈, 3=脊柱, 4=尾根
        self.num_agent_types = 5
        self.agent_type_embed = nn.Embedding(self.num_agent_types, dim)

        # 关键点到类型的映射（按你的索引顺序，示例：右/左后爪, 右/左前爪, 尾根, 头, 颈, 脊柱）
        # 例如： [RR_paw, LR_paw, RF_paw, LF_paw, tail_root, head, neck, spine]
        if self.model_cfg.DATA_TYPE == 'rat':
            self.register_buffer(
                "kp_type_ids",
                torch.tensor([0, 0, 1, 1, 4, 2, 2, 3], dtype=torch.long)  # len=8
            )
        elif self.model_cfg.DATA_TYPE == 'babel':
            kp_type_ids = torch.tensor([
                0,  # 0 pelvis
                1, 1,  # 1-2 左/右hip
                0,  # 3 spine1
                1, 1,  # 4-5 左/右 knee
                0,  # 6 spine2
                1, 1,  # 7-8 左/右 ankle
                0,  # 9 spine3
                1, 1,  # 10-11 左/右 foot
                2,  # 12 neck
                3, 3,  # 13-14 左/右 collar
                2,  # 15 head
                3, 3,  # 16-17 左/右 shoulder
                4, 4, 4, 4  # 18-21 左/右 elbow + wrist
            ], dtype=torch.long)
            self.register_buffer("kp_type_ids", kp_type_ids)


    ### polyline encoder MLP
    def build_polyline_encoder(self, in_channels, hidden_dim, num_layers, num_pre_layers=1, out_channels=None):
        ret_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_pre_layers=num_pre_layers,
            out_channels=out_channels
        )

        return ret_polyline_encoder
    
    # def agent_query_embedding(self, index):
    #     '''
    #     Distinguish between team one, team two and ball. High level PE
    #     One team is at index 0-5
    #     Another team is at index 5-10
    #     Ball is at index 10
    #     '''
    #     team_one_query = self.team_one_query_embedding(index)
    #     team_two_query = self.team_two_query_embedding(index)
    #     ball_query = self.ball_query_embedding(index)
    #     agent_query = torch.cat([team_one_query.repeat(5,1), team_two_query.repeat(5,1), ball_query], dim=0)
    #     return agent_query # [A, D]

    def agent_query_embedding(self, A, device):
        idx = torch.arange(A, device=device)  # [A]
        index_q = self.agent_index_embed(idx)  # [A, D]

        # 截断以匹配 A（以防未来支持可变 agent 数）
        type_ids = self.kp_type_ids[:A].to(device)  # [A]
        type_q = self.agent_type_embed(type_ids)  # [A, D]

        return index_q + type_q  # [A, D]


    def forward(self, past_traj):
        """
        Args: [Batch size, Number of agents, Number of time frames, 6]

        """
        # print("past_traj.device = {}".format(past_traj.device))
        past_traj_mask = torch.ones_like(past_traj[..., 0], dtype=torch.bool).to(past_traj.device)

        self.agent_polyline_encoder.to(device=past_traj.device)
        # print("aaa agent_polyline_encoder first layer device:", next(self.agent_polyline_encoder.parameters()).device)
        obj_polylines_feature = self.agent_polyline_encoder(past_traj, past_traj_mask)  # (num_center_objects, num_objects, C)
        A = obj_polylines_feature.shape[1]
        device = past_traj.device
        # print("123321 past_traj.device = {}".format(device))
        ### use positional encoding pm A
        # pos_encoding = self.pos_encoding(torch.arange(obj_polylines_feature.shape[1]).to(past_traj.device)) #[A, D]
        pos_encoding = self.pos_encoding(torch.arange(A, device=device))

        ### enforce another positional encoding on A earlier here. Disable this before running the ablation
        # agent_query = self.agent_query_embedding(torch.arange(1).to(past_traj.device)) #[A, D]
        agent_query = self.agent_query_embedding(A, device)

        # print("agent_query shape = {}".format(agent_query.shape))
        # print("pos_encoding shape = {}".format(pos_encoding.shape))
        pos_encoding = self.mlp_pe(torch.cat([agent_query, pos_encoding], dim=-1)) #[A, D]
        # pos_encoding = self.mlp_pe(agent_query) #[A, D]

        obj_polylines_feature += pos_encoding.unsqueeze(0) #[B, A, D]
        encoder_out = self.transformer_encoder(obj_polylines_feature)
        
        return encoder_out  
