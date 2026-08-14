"""为 LineRCA 实体定位构造三类可观测证据特征。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.dataprocess import TECH, _observable_features
from model.data import _topology
from model.dynotears import fit_dynotears


def build_localization_features(
    path: str | Path,
    causes: list[str],
    window: int = 8,
    seed: int = 42,
    dynotears_steps: int = 30,
):
    """构造 topology、robust、consistency 三组实体定位特征。

    输入只含可观测测量；干预元数据仅用于划分数据集和生成监督标签，不会进入特征。
    返回 frame、类型标签、三组特征、过滤样本数和动态图边数。
    """
    raw = pd.read_csv(path)
    frame, observable = _observable_features(raw)
    invisible = (
        (frame.technical_root_cause.ne("normal") & frame.technical_severity.le(0.02))
        | (frame.ntl_root_cause.ne("normal") & frame.ntl_severity.le(0.02))
    )
    frame = frame.loc[~invisible].reset_index(drop=True)
    observable = observable.loc[~invisible].reset_index(drop=True)

    # 节点基础量为母线电压和计量负荷，形状 [快照数, 33, 2]。
    voltage = frame[[f"bus_{i:02d}_vm_pu" for i in range(33)]].to_numpy(float)
    meter = frame[[f"bus_{i:02d}_metered_load_kw" for i in range(33)]].to_numpy(float)
    node = np.stack([voltage, meter], axis=-1)

    # topology 证据同时包含物理邻域传播和 DyNOTEARS 因果传播结果。
    physical, _ = _topology(33, 37)
    physical = physical / (physical.sum(axis=1, keepdims=True) + 1e-9)
    train_mask = frame.split.eq("train").to_numpy()
    dynamic, _ = fit_dynotears(
        node[train_mask].mean(axis=-1), steps=dynotears_steps, seed=seed
    )
    causal = dynamic.numpy()
    causal = causal / (causal.sum(axis=1, keepdims=True) + 1e-9)

    local = np.einsum("ij,tjf->tif", physical, node)
    propagated = np.einsum("ij,tjf->tif", causal, node)
    graph = np.c_[
        node.reshape(len(frame), -1),
        local.reshape(len(frame), -1),
        (node - local).reshape(len(frame), -1),
        propagated.reshape(len(frame), -1),
    ]
    line_columns = [
        column
        for column in frame
        if column.startswith("line_")
        and column.endswith(("_loading_pct", "_loss_kw", "_in_service"))
    ]
    topology = np.c_[graph, frame[line_columns].to_numpy(float)]

    # robust 证据在每个事件内部计算滚动中位数、MAD 和一阶差分，绝不跨事件。
    groups = frame.groupby(["grid_id", "event_id"], sort=False)
    robust_parts = []
    for values in (voltage, meter):
        z_score = np.zeros_like(values)
        delta = np.zeros_like(values)
        for indices in groups.groups.values():
            ids = np.asarray(list(indices), dtype=int)
            segment = values[ids]
            rolling = pd.DataFrame(segment).rolling(window, min_periods=1)
            center = rolling.median().to_numpy()
            mad = (
                pd.DataFrame(np.abs(segment - center))
                .rolling(window, min_periods=1)
                .median()
                .to_numpy()
            )
            z_score[ids] = (segment - center) / (1.4826 * np.maximum(mad, 1e-6))
            delta[ids] = np.vstack(
                [np.zeros((1, segment.shape[1])), np.diff(segment, axis=0)]
            )
        robust_parts.extend([np.maximum(z_score, 0), np.maximum(-z_score, 0), np.abs(delta)])

    # consistency 来自泄漏受控的物理一致性量；三组特征最终用于条件实体分类器。
    features = {
        "topology": topology,
        "robust": np.concatenate(robust_parts, axis=1),
        "consistency": observable.drop(columns="_row").to_numpy(float),
    }
    targets = np.zeros((len(frame), len(causes)), dtype=np.int8)
    for index, cause in enumerate(causes):
        source = frame.technical_root_cause if cause in TECH else frame.ntl_root_cause
        targets[:, index] = source.eq(cause)
    return frame, targets, features, int(invisible.sum()), int(dynamic.sum() - 33)
