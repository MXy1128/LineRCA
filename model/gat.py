"""两层 GAT、内嵌 MCE、MAD-ICD 与单参数 CEAF。"""
from __future__ import annotations

import math
import torch
from torch import nn

from .ceaf import AdaptiveEvidenceFusion
from .icd import build_icd
from .mce import MeasurementConsistencyBranch


class GraphAttentionLayer(nn.Module):
    """只沿给定邻接矩阵传播的多头图注意力层。"""

    def __init__(self, input_dim, output_dim, heads=4, dropout=0.1):
        super().__init__()
        if output_dim % heads:
            raise ValueError("hidden_dim must be divisible by gat_heads")
        self.heads = heads
        self.head_dim = output_dim // heads
        self.q = nn.Linear(input_dim, output_dim, bias=False)
        self.k = nn.Linear(input_dim, output_dim, bias=False)
        self.v = nn.Linear(input_dim, output_dim, bias=False)
        self.out = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacency):
        batch, nodes, _ = x.shape

        def split_heads(value):
            return value.view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)

        query, key, value = split_heads(self.q(x)), split_heads(self.k(x)), split_heads(self.v(x))
        scores = query @ key.transpose(-1, -2) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~adjacency.bool()[None, None], torch.finfo(scores.dtype).min
        )
        attention = self.dropout(scores.softmax(dim=-1))
        message = (attention @ value).transpose(1, 2).reshape(batch, nodes, -1)
        projected = self.out(message)
        return self.norm(projected + x if x.shape[-1] == projected.shape[-1] else projected)


class LineRCA(nn.Module):
    """当前论文模型。

    拓扑路：两层 GAT；个体路：基于 Median/MAD 的 ICD；最终由只有
    一个标量 gamma 的 CEAF 进行两路融合。
    """

    def __init__(
        self,
        num_nodes,
        node_dim,
        consistency_dim,
        num_causes,
        hidden_dim=64,
        heads=4,
        layers=2,
        dropout=0.15,
        icd_method="dm_bocd",
        icd_hazard=1.0 / 50.0,
    ):
        super().__init__()
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, hidden_dim) * 0.02)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim * 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gat = nn.ModuleList(
            [GraphAttentionLayer(hidden_dim, hidden_dim, heads, dropout) for _ in range(layers)]
        )
        self.gat_head = nn.Sequential(
            nn.Linear(hidden_dim * num_nodes, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_causes),
        )
        self.mce = MeasurementConsistencyBranch(consistency_dim, hidden_dim, num_causes)
        self.icd_method = icd_method
        self.icd = build_icd(
            icd_method,
            num_nodes,
            node_dim,
            hidden_dim,
            num_causes,
            hazard=icd_hazard,
        )
        self.fusion = AdaptiveEvidenceFusion()

    def forward(self, node_window, consistency_window, adjacency, ablation="full"):
        """返回融合结果和各模块证据。

        node_window 为 [B,T,N,F]，consistency_window 为 [B,T,C]。
        weights 为 [B,2]，依次是拓扑路(GAT+MCE)和个体路(ICD)的权重。
        """
        batch = node_window.shape[0]
        current = node_window[:, -1]
        mean = node_window.mean(dim=1)
        std = node_window.std(dim=1, unbiased=False)
        trend = node_window[:, -1] - node_window[:, 0]
        hidden = self.node_encoder(torch.cat([current, mean, std, trend], dim=-1))
        hidden = hidden + self.node_embedding[None]
        if ablation != "no_gat":
            for layer in self.gat:
                hidden = layer(hidden, adjacency)
        gat_logits = self.gat_head(hidden.flatten(start_dim=1))

        mce_logits = self.mce(consistency_window)
        if ablation == "no_gat":
            gat_logits = torch.zeros_like(gat_logits)
            mce_logits = torch.zeros_like(mce_logits)
            topological = torch.zeros_like(gat_logits)
        else:
            topological = 0.5 * (gat_logits + mce_logits)

        # A structural ablation must remove the branch from the forward graph,
        # rather than merely zeroing its already-computed output afterwards.
        individual = (
            torch.zeros_like(topological)
            if ablation == "no_icd"
            else self.icd(node_window)
        )
        if ablation == "no_gat":
            logits = individual
            weights = individual.new_tensor([0.0, 1.0]).expand(batch, -1)
        elif ablation == "no_icd":
            logits = topological
            weights = topological.new_tensor([1.0, 0.0]).expand(batch, -1)
        elif ablation == "no_ceaf":
            logits = 0.5 * (topological + individual)
            weights = topological.new_tensor([0.5, 0.5]).expand(batch, -1)
        else:
            logits, weights = self.fusion(topological, individual)

        return {
            "logits": logits,
            "weights": weights,
            "topology": topological,
            "gat": gat_logits,
            "mce": mce_logits,
            "icd": individual,
        }
