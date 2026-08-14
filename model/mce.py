"""MCE（Measurement Consistency Evidence）测量一致性证据分支。"""
import torch
from torch import nn


class MeasurementConsistencyBranch(nn.Module):
    """汇聚能量平衡、计量差额、电压和线路状态等一致性特征。

    输入 [B,T,C]，输出 [B,根因类型数]。
    """
    def __init__(self, input_dim, hidden_dim, num_causes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 5, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_causes),
        )

    def forward(self, x):
        # 当前状态表示故障时刻；均值/标准差/最大值/趋势描述窗口背景与变化。
        mean = x.mean(1)
        std = x.std(1, unbiased=False)
        maximum = x.abs().amax(1)
        trend = x[:, -1] - x[:, 0]
        return self.net(torch.cat([x[:, -1], mean, std, maximum, trend], dim=-1))
