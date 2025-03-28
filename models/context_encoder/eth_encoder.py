import numpy as np
import torch
import torch.nn as nn


from models.utils import polyline_encoder
from models.context_encoder.mtr_encoder import SinusoidalPosEmb
from einops import rearrange
import math


class SocialTransformer(nn.Module):
    def __init__(self, in_dim=48, hidden_dim=256, out_dim=128):
        super(SocialTransformer, self).__init__()
        self.encode_past = nn.Linear(in_dim, hidden_dim, bias=False)
        self.layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)
        self.mlp_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, past_traj, mask):
        """
        @param past_traj: [B, A, P, D]
        @param mask:      [B, A] or None
        """
        B, A, P, D = past_traj.shape
        # past_traj = rearrange(past_traj, 'b a p d -> (b a) p d')
        # h_feat = self.encode_past(past_traj.reshape(B * A, -1)).unsqueeze(1)  # [B*A, 1, D]

        past_traj = rearrange(past_traj, 'b a p d -> b a (p d)')
        h_feat = self.encode_past(past_traj)    # [B, A, D]

        h_feat_ = self.transformer_encoder(h_feat, mask=mask)

        h_feat = h_feat + h_feat_
        h_feat = self.mlp_out(h_feat)           # [B, A, D]

        return h_feat
    

class ETHEncoder(nn.Module):
    def __init__(self, config, use_pre_norm):
        super().__init__()
        self.model_cfg = config
        dim = self.model_cfg.D_MODEL
        
        ### build social encoder
        self.agent_social_encoder = SocialTransformer(in_dim=48, hidden_dim=256, out_dim=dim)
        
        # Positional encoding
        self.pos_encoding = nn.Sequential(
                SinusoidalPosEmb(dim, theta = 10000),
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            )
        self.agent_query_embedding = nn.Embedding(self.model_cfg.AGENTS, dim)
        self.mlp_pe = nn.Sequential(
            nn.Linear(2*dim, dim),
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
        
    ### polyline encoder MLP PointNet [B, A, D]
    def build_polyline_encoder(self, in_channels, hidden_dim, num_layers, num_pre_layers=1, out_channels=None):
        ret_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_pre_layers=num_pre_layers,
            out_channels=out_channels
        )
        return ret_polyline_encoder
    

    def forward(self, past_traj):
        """
        Args: [Batch size, Number of agents, Number of time frames, 6]
        """
        
        B, A, P, D = past_traj.shape
        agent_feature = self.agent_social_encoder(past_traj, mask=None)  # [B, A, D]

        ### use positional encoding
        pos_encoding = self.pos_encoding(torch.arange(agent_feature.shape[1]).to(past_traj.device))         # [A, D]

        ### enforce positional encoding earlier here
        agent_query = self.agent_query_embedding(torch.arange(self.model_cfg.AGENTS).to(past_traj.device))  # [A, D]

        pos_encoding = self.mlp_pe(torch.cat([agent_query, pos_encoding], dim=-1)) # [A, D]

        agent_feature += pos_encoding.unsqueeze(0)                              # [B, A, D]
        encoder_out = self.transformer_encoder(agent_feature)
        
        return encoder_out 
    