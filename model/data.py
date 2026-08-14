"""把 Pandapower 快照数据转换为 GAT 可使用的事件内时间窗口。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

TECH_CAUSES = ["topology_reconfiguration", "poor_power_factor", "overload",
               "line_resistance_increase", "low_source_voltage", "reverse_power_flow"]
NTL_CAUSES = ["meter_bypass", "mean_replacement", "meter_bias", "uniform_scaling",
              "intermittent_dropout", "peak_period_theft"]
CAUSES = TECH_CAUSES + NTL_CAUSES


@dataclass
class DataBundle:
    """训练入口所需数据的统一容器。

    loaders: train/validation/test 三个 DataLoader；
    physical_adjacency: [节点数, 节点数] 的物理邻接矩阵；
    causal_series: 仅由训练集构造，供 DyNOTEARS 学习动态图。
    """
    loaders: dict
    labels: list[str]
    node_dim: int
    consistency_dim: int
    num_nodes: int
    physical_adjacency: torch.Tensor
    causal_series: np.ndarray
    split_sizes: dict


class WindowDataset(Dataset):
    """单个样本包含节点时间窗、MCE 时间窗、标签和当前行号。"""
    def __init__(self, node, consistency, labels, rows):
        self.node = torch.as_tensor(node, dtype=torch.float32)
        self.consistency = torch.as_tensor(consistency, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.rows = rows.reset_index(drop=True)

    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.node[i], self.consistency[i], self.labels[i], i


def _topology(num_nodes: int, num_lines: int):
    """构造 IEEE 33 节点物理邻接矩阵，并保留自环。"""
    # 这里的支路顺序与 pandapower.networks.case33bw 完全一致（从 0 编号）。
    # 直接保存端点可以避免训练时加载 pandapower 及其潮流求解依赖。
    case33bw_ends = [(i, i+1) for i in range(17)] + [(1,18),(18,19),(19,20),(20,21),
        (2,22),(22,23),(23,24),(5,25),(25,26),(26,27),(27,28),(28,29),(29,30),(30,31),(31,32),
        (20,7),(8,14),(11,21),(17,32),(24,28)]
    ends = case33bw_ends if num_nodes == 33 else [(i, i + 1) for i in range(num_nodes - 1)]
    ends = ends[:num_lines]
    adj = np.eye(num_nodes, dtype=np.float32)
    for a, b in ends:
        if a < num_nodes and b < num_nodes:
            adj[a, b] = adj[b, a] = 1.0
    return adj, ends


def _robust_fit(x):
    """用训练集的中位数和 MAD 拟合抗异常值标准化参数。"""
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0)
    return med, np.maximum(1.4826 * mad, 1e-6)


def load_data(csv_path: str | Path, window_size=12, batch_size=64, seed=42):
    """加载单个原始 CSV 并返回 :class:`DataBundle`。

    关键形状：
    node window       [样本, 时间窗, 节点, 4]
    consistency window[样本, 时间窗, 一致性特征数]
    label             [样本, 12]

    标准化参数只由训练集估计，时间窗口不跨 event_id，防止数据泄漏。
    """
    df = (pd.read_csv(csv_path, parse_dates=["timestamp"])
          .sort_values(["grid_id", "timestamp"]).reset_index(drop=True))
    # 严重度不超过 0.02 的注入在测量上不可可靠辨识，与公平基线采用同一过滤规则。
    invisible = ((df["technical_root_cause"].ne("normal") & df["technical_severity"].le(.02)) |
                 (df["ntl_root_cause"].ne("normal") & df["ntl_severity"].le(.02)))
    df = df.loc[~invisible].reset_index(drop=True)
    bus_ids = sorted(int(re.search(r"bus_(\d+)_vm_pu", c).group(1))
                     for c in df if re.fullmatch(r"bus_\d+_vm_pu", c))
    line_ids = sorted(int(re.search(r"line_(\d+)_loading_pct", c).group(1))
                      for c in df if re.fullmatch(r"line_\d+_loading_pct", c))
    n = len(bus_ids)
    physical, ends = _topology(n, len(line_ids))

    # 每个母线的 4 维节点特征：电压、计量负荷、相邻线路平均负载率和平均损耗。
    voltage = df[[f"bus_{i:02d}_vm_pu" for i in bus_ids]].to_numpy(float)
    metered = df[[f"bus_{i:02d}_metered_load_kw" for i in bus_ids]].to_numpy(float)
    inc_loading = np.zeros_like(voltage)
    inc_loss = np.zeros_like(voltage)
    for line, (a, b) in zip(line_ids, ends):
        loading = df[f"line_{line:02d}_loading_pct"].to_numpy(float)
        loss = df[f"line_{line:02d}_loss_kw"].to_numpy(float)
        inc_loading[:, [a, b]] += loading[:, None]
        inc_loss[:, [a, b]] += loss[:, None]
    degree = np.maximum(physical.sum(1) - 1, 1)
    inc_loading /= degree[None, :]
    inc_loss /= degree[None, :]
    node_raw = np.stack([voltage, metered, inc_loading, inc_loss], axis=-1)

    # MCE 只使用运行中可观测/可估计的量，刻意排除仿真器中的真实负荷和干预参数。
    # 此处仍是一条原始快照对应一个样本，不进行候选根因展开。
    from data.dataprocess import _observable_features
    _, observable = _observable_features(df)
    consistency_raw = observable.drop(columns="_row").to_numpy(float)

    # 中位数和 MAD 只在训练集拟合，验证集和测试集绝不参与尺度估计。
    train_mask = df["split"].eq("train").to_numpy()
    node_med, node_scale = _robust_fit(node_raw[train_mask].reshape(-1, node_raw.shape[-1]))
    con_med, con_scale = _robust_fit(consistency_raw[train_mask])
    node_raw = (node_raw - node_med) / node_scale
    consistency_raw = (consistency_raw - con_med) / con_scale

    # 多热标签允许 mixed 数据中的同一时刻同时存在技术性和非技术性根因。
    ys = np.zeros((len(df), len(CAUSES)), np.float32)
    for j, cause in enumerate(CAUSES):
        col = "technical_root_cause" if cause in TECH_CAUSES else "ntl_root_cause"
        ys[:, j] = df[col].eq(cause).to_numpy(np.float32)

    datasets, sizes = {}, {}
    for split in ("train", "validation", "test"):
        xs, cs, ls, rs = [], [], [], []
        # 按 grid_id + event_id 分组，时间窗不会跨越两次不同的故障注入事件。
        for _, idx in df[df["split"].eq(split)].groupby(["grid_id", "event_id"], sort=False).groups.items():
            idx = np.asarray(list(idx), dtype=int)
            for stop in range(len(idx)):
                begin = max(0, stop - window_size + 1)
                win = idx[begin:stop + 1]
                if len(win) < window_size:
                    win = np.concatenate([np.repeat(win[:1], window_size-len(win)), win])
                xs.append(node_raw[win]); cs.append(consistency_raw[win]); ls.append(ys[idx[stop]])
                rs.append(df.iloc[idx[stop]])
        datasets[split] = WindowDataset(np.asarray(xs), np.asarray(cs), np.asarray(ls), pd.DataFrame(rs))
        sizes[split] = len(datasets[split])
    gen = torch.Generator().manual_seed(seed)
    loaders = {s: DataLoader(ds, batch_size=batch_size, shuffle=s == "train", generator=gen)
               for s, ds in datasets.items()}
    # DyNOTEARS 输入为训练集每个节点的电压/计量负荷综合序列 [时间, 节点]。
    causal_series = node_raw[train_mask, :, :2].mean(axis=-1)
    return DataBundle(loaders, CAUSES, node_raw.shape[-1], consistency_raw.shape[-1], n,
                      torch.from_numpy(physical), causal_series, sizes)
