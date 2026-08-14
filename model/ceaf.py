"""CEAF：使用一个可学习参数融合拓扑证据和个体证据。"""
import torch
from torch import nn


class AdaptiveEvidenceFusion(nn.Module):
    """实现 gamma*ICD + (1-gamma)*(GAT+MCE)。

    模块只有一个可训练标量 ``gamma_logit``；Sigmoid 后得到 0~1 的 gamma。
    第二个权重由 1-gamma 自动确定，因此不存在额外门控网络。
    """

    def __init__(self):
        super().__init__()
        self.gamma_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, topological, individual):
        gamma = torch.sigmoid(self.gamma_logit)
        fused = gamma * individual + (1.0 - gamma) * topological
        weights = torch.stack([1.0 - gamma, gamma]).expand(len(topological), -1)
        return fused, weights
