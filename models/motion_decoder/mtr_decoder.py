import copy
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import torch


def modulate(x, shift, scale):
    if len(x.shape) == 3 and len(shift.shape) == 2:
        # [B, K, D] + [B, D]
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    elif len(x.shape) == len(shift.shape) == 3:
        # [B, K, D] + [B, K, D]
        return x * (1 + scale) + shift
    elif len(x.shape) == 4 and len(shift.shape) == 2:
        # [B, K, A, D] + [B, D]
        return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)
    elif len(x.shape) == len(shift.shape) == 4:
        # [B, K, A, D] + [B, K, A, D]
        return x * (1 + scale) + shift
    else:
        raise ValueError("Invalid shapes to modulate")
    

class MTRDecoder(nn.Module):
    def __init__(self, config, use_pre_norm, use_adaln=True):
        super().__init__()
        self.num_blocks = config.get('NUM_DECODER_BLOCKS', 2)
        self.self_attn_K = nn.ModuleList([])
        self.self_attn_A = nn.ModuleList([])
        template_encoder = nn.TransformerEncoderLayer(d_model=config.D_MODEL, 
                                                      dropout=config.get('DROPOUT_OF_ATTN', 0.1),
                                                      nhead=config.NUM_ATTN_HEAD, 
                                                      dim_feedforward=config.D_MODEL * 4, 
                                                      norm_first=use_pre_norm,
                                                      batch_first=True)
        self.use_adaln = use_adaln

        if use_adaln:
            template_adaln = nn.Sequential(nn.SiLU(),
                                        nn.Linear(config.D_MODEL, 2 * config.D_MODEL, bias=True))
            
            self.t_adaLN = nn.ModuleList([])

        for _ in range(self.num_blocks):
            self.self_attn_K.append(copy.deepcopy(template_encoder))
            self.self_attn_A.append(copy.deepcopy(template_encoder))

            if use_adaln:
                self.t_adaLN.append(copy.deepcopy(template_adaln))

                # zero initialization parameters of adaln
                nn.init.constant_(self.t_adaLN[-1][-1].weight, 0)
                nn.init.constant_(self.t_adaLN[-1][-1].bias, 0)

        
    def forward(self, query_token, time_emb=None):
        """
        @param query_token: [B, K, A, D]
        @param time_emb: [B, D]
        """
        B, K, A = query_token.shape[:3]
        cur_query = query_token
        
        for i in range(self.num_blocks):
            # if self.use_adaln:
            #     # time modulation
            #     shift, scale = self.t_adaLN[i](time_emb).chunk(2, dim=-1)
            #     cur_query = modulate(cur_query, shift, scale)       # [B, K, A, D]

            # K-to-K self-attention
            # print("cur_query shape 1 = {}".format(cur_query.shape))

            cur_query = rearrange(query_token, 'b k a t d -> (b a t) k d')
            # cur_query = rearrange(query_token, 'b k a d -> (b a) k d')
            cur_query = self.self_attn_K[i](cur_query)

            # A-to-A self-attention
            cur_query = rearrange(cur_query, '(b a t) k d -> (b k t) a d', b=B, a=A, k=K)
            # cur_query = rearrange(cur_query, '(b a) k d -> (b k) a d', b=B, a=A, k=K)
            cur_query = self.self_attn_A[i](cur_query)

            # reshape
            cur_query = rearrange(cur_query, '(b k t) a d -> b k a t d', b=B, a=A, k=K)

        return cur_query
    
