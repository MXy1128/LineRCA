"""带 NOTEARS 无环约束的稀疏滞后动态因果图估计器。"""
from __future__ import annotations
import numpy as np
import torch


def fit_dynotears(series, lags=2, steps=120, lr=0.02, l1=0.01, quantile=0.85, seed=42):
    """根据滞后关系估计有向动态因果图。

    输入 series 的形状为 [时间, 节点]。模型学习 W[k, i, j]，表示节点 i
    在 t-k 时刻对节点 j 在 t 时刻的影响。各滞后矩阵绝对值求和得到边强度，
    trace(exp(W²))-d 是 NOTEARS 的可微无环约束。最终仅保留强度最高的一部分边。
    """
    torch.manual_seed(seed)
    x = torch.as_tensor(np.asarray(series), dtype=torch.float32)
    x = (x - x.mean(0)) / x.std(0).clamp_min(1e-6)
    # target 是当前时刻，lagged 依次保存前 1...lags 个时刻。
    target = x[lags:]
    lagged = torch.stack([x[lags-k:-k] for k in range(1, lags + 1)], dim=1)
    # 某些第三方因果库会修改 PyTorch 默认精度，显式绑定 dtype 保证运行顺序不影响结果。
    w = torch.zeros(lags, x.shape[1], x.shape[1], dtype=x.dtype, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    rho, alpha = 1.0, 0.0
    for step in range(steps):
        opt.zero_grad()
        # 预测误差 + L1 稀疏项 + 增广拉格朗日无环约束。
        pred = torch.einsum("tld,ldj->tj", lagged, w)
        strength = w.abs().sum(0)
        h = torch.trace(torch.matrix_exp(strength.square())) - x.shape[1]
        loss = (pred-target).square().mean() + l1*w.abs().mean() + alpha*h + 0.5*rho*h*h
        loss.backward(); opt.step()
        if (step + 1) % max(steps // 4, 1) == 0:
            alpha += rho * float(h.detach()); rho = min(rho * 2.0, 1e5)
    score = w.detach().abs().sum(0).numpy()
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    # 按分位数稀疏化，并加入自环以便 GAT 保留节点自身信息。
    positive = score[score > 1e-12]
    threshold = np.quantile(positive, quantile) if positive.size else np.inf
    adjacency = (score >= threshold).astype(np.float32)
    np.fill_diagonal(adjacency, 1.0)
    return torch.from_numpy(adjacency), score
