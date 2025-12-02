import torch
import torch.nn as nn
from einops import repeat, rearrange

class Z0Encoder(nn.Module):
    """
    将历史大鼠骨架 + 历史刺激编码为初始隐变量 z0。

    默认假设：
        - hist_kp: [B, Th, J, D]  (J=关键点数, D=2 或 3)
        - hist_stim: [B, Th, stim_dim]

    你可以根据自己的 x_data 维度轻微调整 reshape 部分的代码。
    """

    def __init__(
        self,
        num_keypoints: int,
        kp_dim: int,
        stim_dim: int,
        hidden_dim: int,
        z_dim: int,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.kp_dim = kp_dim
        self.stim_dim = stim_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        in_dim = num_keypoints * kp_dim + stim_dim

        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_dim, z_dim)


    def forward(
        self,
        hist_kp: torch.Tensor,
        hist_stim: torch.Tensor,
    ) -> torch.Tensor:
        """
        hist_kp:   [B, Th, J, D]
        hist_stim: [B, Th, stim_dim]
        return:
            z0: [B, z_dim]
        """
        B, J, Th, D = hist_kp.shape
        _, Th2, stim_dim = hist_stim.shape
        # print("hist_kp shape = {}".format(hist_kp.shape))
        # print("hist_stim shape = {}".format(hist_stim.shape))
        assert Th == Th2, "hist_kp 和 hist_stim 的时间长度不一致"

        # 把 [J, D] 展平成一维特征
        # kp_feat = hist_kp.reshape(B, Th, J * D)            # [B, Th, J*D]
        kp_feat = rearrange(hist_kp, 'b j Th d -> b Th (j d)')
        # print("kp_feat shape = {}".format(kp_feat.shape))
        # print("hist_stim shape = {}".format(hist_stim.shape))

        x = torch.cat([kp_feat, hist_stim], dim=-1)        # [B, Th, J*D+stim_dim]
        # print("x shape = {}".format(x.shape))
        # 为什么有个0向量？
        h0 = torch.zeros(
            1,
            B,
            self.hidden_dim,
            device=hist_kp.device,
            dtype=hist_kp.dtype,
        )
        out, h_last = self.gru(x, h0)                      # h_last: [1, B, H]
        z0 = self.linear(h_last.squeeze(0))                # [B, z_dim]
        return z0
