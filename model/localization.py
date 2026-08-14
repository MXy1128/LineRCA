"""LineRCA 的“根因类型 → 具体设备”条件实体定位层。"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier


TECHNICAL_SCHEMA = {
    "overload": ("load", range(32)),
    "line_resistance_increase": ("line", range(32)),
    "low_source_voltage": ("ext_grid", range(1)),
    "poor_power_factor": ("load", range(32)),
    "reverse_power_flow": ("bus", range(1, 33)),
    "topology_reconfiguration": ("line", range(32, 37)),
}


def candidate_key(cause: str, kind: str, index: int) -> str:
    """生成全局唯一候选名，例如 overload@load_03。"""
    return f"{cause}@{kind}_{int(index):02d}"


def build_candidate_schema(dataset: str, causes: list[str]) -> list[str]:
    """枚举数据集中所有允许的“根因类型@设备”候选。"""
    del dataset  # The active cause list already determines the dataset scope.
    candidates = []
    for cause in causes:
        if cause in TECHNICAL_SCHEMA:
            kind, indices = TECHNICAL_SCHEMA[cause]
            candidates.extend(candidate_key(cause, kind, index) for index in indices)
        else:
            candidates.extend(candidate_key(cause, "bus", index) for index in range(1, 33))
    return candidates


def extract_true_entities(frame: pd.DataFrame) -> list[list[str]]:
    """从标签列读取真实根因实体；mixed 样本可以同时返回两个实体。"""
    entities = []
    for row in frame.itertuples(index=False):
        roots = []
        if row.technical_root_cause != "normal":
            roots.append(
                candidate_key(
                    row.technical_root_cause,
                    row.technical_root_element_type,
                    row.technical_root_element_index,
                )
            )
        if row.ntl_root_cause != "normal":
            roots.append(candidate_key(row.ntl_root_cause, "bus", row.ntl_root_bus))
        entities.append(roots)
    return entities


def fit_entity_localizers(
    features: np.ndarray,
    frame: pd.DataFrame,
    causes: list[str],
    train_mask: np.ndarray,
    seed: int,
    trees: int,
    logger: logging.Logger,
):
    """为每种根因分别训练 P(实体 | 根因类型) 的 ExtraTrees 定位器。

    注意：神经网络负责根因类型评分；这里的树只在已知类型内部区分具体设备。
    所有定位器仅使用 train_mask 指定的训练样本。
    """
    localizers = {}
    for offset, cause in enumerate(causes):
        if cause in TECHNICAL_SCHEMA:
            mask = train_mask & frame.technical_root_cause.eq(cause).to_numpy()
            target = frame.loc[mask, "technical_root_element_index"].astype(int).to_numpy()
        else:
            mask = train_mask & frame.ntl_root_cause.eq(cause).to_numpy()
            target = frame.loc[mask, "ntl_root_bus"].astype(int).to_numpy()

        # 某些根因只能发生在唯一实体，例如低电源电压只能对应 ext_grid_00。
        if len(np.unique(target)) == 1:
            localizers[cause] = int(target[0])
            logger.info("ENTITY/%s | deterministic=%d | samples=%d", cause, target[0], len(target))
            continue
        model = ExtraTreesClassifier(
            n_estimators=trees,
            min_samples_leaf=1,
            max_features=0.7,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed + offset,
        )
        model.fit(features[mask], target)
        localizers[cause] = model
        logger.info("ENTITY/%s | samples=%d | classes=%d", cause, len(target), len(model.classes_))
    return localizers


def combine_type_and_entity_scores(
    features: np.ndarray,
    type_scores: np.ndarray,
    causes: list[str],
    candidates: list[str],
    localizers: dict,
) -> np.ndarray:
    """计算联合分数 P(type, entity)=P(type)×P(entity|type)。"""
    positions = {candidate: index for index, candidate in enumerate(candidates)}
    scores = np.zeros((len(features), len(candidates)), dtype=np.float32)
    for cause_index, cause in enumerate(causes):
        localizer = localizers[cause]
        if isinstance(localizer, int):
            classes = np.array([localizer])
            probabilities = np.ones((len(features), 1))
        else:
            classes = localizer.classes_.astype(int)
            probabilities = localizer.predict_proba(features)
        kind = TECHNICAL_SCHEMA[cause][0] if cause in TECHNICAL_SCHEMA else "bus"
        for column, entity_index in enumerate(classes):
            candidate = candidate_key(cause, kind, entity_index)
            if candidate in positions:
                scores[:, positions[candidate]] = type_scores[:, cause_index] * probabilities[:, column]
    return scores


def encode_entity_labels(true_entities: list[list[str]], candidates: list[str]) -> np.ndarray:
    """把字符串形式的真实实体转换为排名指标需要的多热矩阵。"""
    positions = {candidate: index for index, candidate in enumerate(candidates)}
    labels = np.zeros((len(true_entities), len(candidates)), dtype=np.int8)
    for row_index, roots in enumerate(true_entities):
        for root in roots:
            if root in positions:
                labels[row_index, positions[root]] = 1
    return labels
