"""LineRCA 的唯一训练入口。

完整流程：读取一个 Pandapower CSV → 构造时间窗 → 学习 DyNOTEARS 图 →
训练两层 GAT+ICD+CEAF → 根因类型打分 → 具体设备实体定位 → 计算排名指标。
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.dataprocess import CAUSES
from metrics.ranking import localization_metrics
from model.data import load_data
from model.dynotears import fit_dynotears
from model.gat import LineRCA
from model.localization import (
    build_candidate_schema,
    combine_type_and_entity_scores,
    encode_entity_labels,
    extract_true_entities,
    fit_entity_localizers,
)
from model.preprocessing import build_localization_features


ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = {
    "technical": "technical_line_loss.csv",
    "non_technical": "non_technical_line_loss.csv",
    "mixed": "mixed_line_loss.csv",
}
ARCHITECTURE = "{layers}GAT+DyNOTEARS+(GAT-MCE)+{icd}-ICD+1P-CEAF"


def parse_args() -> argparse.Namespace:
    """定义实验参数；默认值就是正式实验当前使用的配置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATA_FILES, default="mixed")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "simulated_pandapower_v2",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument(
        "--icd-method", choices=("dm_bocd", "bocd", "mad"), default="dm_bocd",
        help="ICD实现：dm_bocd为扩散分数匹配鲁棒BOCD，bocd为标准BOCD，mad为稳健基线",
    )
    parser.add_argument(
        "--ablation",
        choices=("full", "no_dynotears", "no_gat", "no_icd", "no_ceaf"),
        default="full",
        help="单模块消融；full 为完整模型",
    )
    parser.add_argument("--dynotears-steps", type=int, default=120)
    parser.add_argument("--entity-trees", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def causes_for(dataset: str) -> list[str]:
    """根据数据集选择有效根因类型：技术 6 类、非技术 6 类或全部 12 类。"""
    if dataset == "technical":
        return CAUSES[:6]
    if dataset == "non_technical":
        return CAUSES[6:]
    return CAUSES


def make_logger(path: Path) -> logging.Logger:
    """同时向日志文件和控制台输出，日志中保留每个 epoch 的信息。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("line_rca_gat")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


@torch.no_grad()
def predict(model, loader, adjacency, device, ablation="full"):
    """批量推理。

    返回：根因类型概率 [样本数, 根因数]、三个分支的融合权重
    [样本数, 3]，以及样本在当前 Dataset 中的行号。
    """
    model.eval()
    score_batches, weight_batches, row_batches = [], [], []
    for node_features, consistency, _, row_indices in loader:
        output = model(
            node_features.to(device), consistency.to(device), adjacency, ablation
        )
        score_batches.append(torch.sigmoid(output["logits"]).cpu().numpy())
        weight_batches.append(output["weights"].cpu().numpy())
        row_batches.append(row_indices.numpy())
    return (
        np.concatenate(score_batches),
        np.concatenate(weight_batches),
        np.concatenate(row_batches),
    )


def train_type_model(model, bundle, adjacency, active_indices, args, device, logger):
    """训练根因类型评分网络，并用验证损失早停。

    总损失 = 融合输出 BCE + 0.2 × 三个分支各自的 BCE。辅助损失保证
    GAT、ICD、MCE 每个分支单独也能提供有效证据，避免融合层只依赖某一支。
    """
    # 正负样本不平衡明显，因此只用训练集计算每类的 BCE 正样本权重。
    positives = torch.zeros(len(active_indices))
    sample_count = 0
    for _, _, labels, _ in bundle.loaders["train"]:
        positives += labels[:, active_indices].sum(axis=0)
        sample_count += len(labels)
    positive_weight = ((sample_count - positives) / positives.clamp_min(1)).clamp_max(20)
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    final_epoch = 0
    for epoch in range(1, args.epochs + 1):
        final_epoch = epoch
        model.train()
        total_loss = 0.0
        for node_features, consistency, labels, _ in bundle.loaders["train"]:
            node_features = node_features.to(device)
            consistency = consistency.to(device)
            labels = labels[:, active_indices].to(device)
            optimizer.zero_grad()
            output = model(node_features, consistency, adjacency, args.ablation)
            loss = loss_function(output["logits"], labels)
            auxiliary_branches = [] if args.ablation == "no_gat" else ["topology"]
            if args.ablation != "no_icd":
                auxiliary_branches.append("icd")
            loss += 0.2 * sum(
                loss_function(output[branch], labels) for branch in auxiliary_branches
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)

        # 验证阶段只评价融合 logits，不参与梯度更新。
        model.eval()
        validation_loss = 0.0
        validation_count = 0
        with torch.no_grad():
            for node_features, consistency, labels, _ in bundle.loaders["validation"]:
                labels = labels[:, active_indices].to(device)
                logits = model(
                    node_features.to(device), consistency.to(device), adjacency,
                    args.ablation,
                )["logits"]
                validation_loss += loss_function(logits, labels).item() * len(labels)
                validation_count += len(labels)
        validation_loss /= validation_count
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        dm_omega = float(model.icd.omega.detach()) if hasattr(model.icd, "omega") else float("nan")
        logger.info(
            "MODEL/EPOCH %d/%d | train_loss=%.6f | val_loss=%.6f | best_val_loss=%.6f | gamma=%.6f | dm_omega=%.6f",
            epoch,
            args.epochs,
            total_loss / sample_count,
            validation_loss,
            best_loss,
            float(torch.sigmoid(model.fusion.gamma_logit).detach()),
            dm_omega,
        )
        # 正式实验至少训练 min_epochs；之后验证损失连续 patience 轮不改善才早停。
        if epoch >= args.min_epochs and stale_epochs >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    return final_epoch, best_epoch


def main() -> None:
    """串联训练、定位、评估和文件保存。"""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    causes = causes_for(args.dataset)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = ROOT / "results" / "gat_entity_localization" / f"{timestamp}_{args.dataset}"
    result_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(ROOT / "logs" / f"gat_entity_localization_{args.dataset}_{timestamp}.log")
    logger.info(
        "RUN/START | dataset=%s | architecture=%s | seed=%d | device=%s",
        args.dataset,
        ARCHITECTURE.format(layers=args.layers, icd=args.icd_method.upper()),
        args.seed,
        device,
    )
    logger.info("RUN/PARAMS | %s", json.dumps(vars(args), default=str))
    start_time = time.perf_counter()

    # 第一阶段：读取时序数据，并学习“物理拓扑 ∪ 动态因果关系”融合图。
    data_path = args.data_dir / DATA_FILES[args.dataset]
    bundle = load_data(data_path, args.window, args.batch_size, args.seed)
    active_indices = [bundle.labels.index(cause) for cause in causes]
    if args.ablation == "no_dynotears":
        dynamic_adjacency = torch.eye(bundle.num_nodes)
        adjacency = bundle.physical_adjacency.float().to(device)
    else:
        dynamic_adjacency, _ = fit_dynotears(
            bundle.causal_series, steps=args.dynotears_steps, seed=args.seed
        )
        adjacency = ((bundle.physical_adjacency + dynamic_adjacency) > 0).float().to(device)
    model = LineRCA(
        bundle.num_nodes,
        bundle.node_dim,
        bundle.consistency_dim,
        len(causes),
        args.hidden,
        args.heads,
        args.layers,
        icd_method=args.icd_method,
    ).to(device)
    parameter_count = sum(value.numel() for value in model.parameters() if value.requires_grad)
    logger.info(
        "MODEL/GRAPH | physical_edges=%d | dynamic_edges=%d | fused_edges=%d",
        int(bundle.physical_adjacency.sum() - bundle.num_nodes),
        int(dynamic_adjacency.sum() - bundle.num_nodes),
        int(adjacency.sum() - bundle.num_nodes),
    )
    logger.info("MODEL/START | params=%d", parameter_count)
    # 第二阶段：网络输出每个样本属于各根因类型的概率。
    final_epoch, best_epoch = train_type_model(
        model, bundle, adjacency, active_indices, args, device, logger
    )

    type_scores, fusion_weights, row_indices = predict(
        model, bundle.loaders["test"], adjacency, device, args.ablation
    )
    test_rows = (
        bundle.loaders["test"].dataset.rows.iloc[row_indices].reset_index(drop=True)
    )

    # 第三阶段：在根因类型内部定位具体线路、母线、负荷或外部电网。
    # 定位器只使用训练集拟合；测试集仅用于最终评价。
    frame, _, evidence, removed, localization_edges = build_localization_features(
        data_path, causes, window=8, seed=args.seed, dynotears_steps=30
    )
    logger.info(
        "DATA/LOCALIZATION | rows=%d | invisible_removed=%d | dynamic_edges=%d",
        len(frame),
        removed,
        localization_edges,
    )
    entity_features = np.c_[
        evidence["topology"], evidence["robust"], evidence["consistency"]
    ]
    train_mask = frame.split.eq("train").to_numpy()
    localizers = fit_entity_localizers(
        entity_features,
        frame,
        causes,
        train_mask,
        args.seed,
        args.entity_trees,
        logger,
    )
    # 两套预处理都来自同一原始 CSV，用三元键将神经网络测试行与定位特征对齐。
    row_lookup = {
        (str(row.timestamp), str(row.grid_id), str(row.event_id)): index
        for index, row in frame.iterrows()
    }
    localization_rows = np.array(
        [
            row_lookup[(str(row.timestamp), str(row.grid_id), str(row.event_id))]
            for row in test_rows.itertuples(index=False)
        ]
    )
    candidates = build_candidate_schema(args.dataset, causes)
    # 联合分数 P(具体实体) = P(根因类型) × P(实体 | 根因类型)。
    entity_scores = combine_type_and_entity_scores(
        entity_features[localization_rows],
        type_scores,
        causes,
        candidates,
        localizers,
    )
    true_entities = extract_true_entities(test_rows)
    entity_labels = encode_entity_labels(true_entities, candidates)
    metrics = localization_metrics(entity_labels, entity_scores)
    result = {
        "Dataset": args.dataset,
        "Model": f"LineRCA-GAT-{args.icd_method.upper()}-ICD",
        "Ablation": args.ablation,
        "Layers": args.layers,
        "Candidates": len(candidates),
        "TestRooted": int((entity_labels.sum(axis=1) > 0).sum()),
        "Epochs": final_epoch,
        "BestEpoch": best_epoch,
        "Params": parameter_count,
        "DmOmega": float(model.icd.omega.detach()) if hasattr(model.icd, "omega") else np.nan,
        "Seconds": time.perf_counter() - start_time,
        **metrics,
    }
    # 保存完整实验产物，确保论文结果可以复查和复现。
    pd.DataFrame([result]).to_csv(result_dir / "metrics.csv", index=False)

    ranking = np.argsort(-entity_scores, axis=1)[:, :10]
    predictions = test_rows[["timestamp", "grid_id", "root_cause", "loss_type"]].copy()
    predictions["true_entities"] = ["+".join(entities) for entities in true_entities]
    for rank in range(10):
        predictions[f"rank_{rank + 1}"] = [candidates[index] for index in ranking[:, rank]]
    predictions[["weight_gat_mce", "weight_icd"]] = fusion_weights
    predictions.to_csv(result_dir / "top10_predictions.csv", index=False)
    torch.save(
        {
            "model": model.state_dict(),
            "adjacency": adjacency.cpu(),
            "causes": causes,
            "ceaf_gamma": float(torch.sigmoid(model.fusion.gamma_logit).detach()),
        },
        result_dir / "best_model.pt",
    )
    logger.info("MODEL/FINAL | %s", json.dumps(result))
    logger.info("RUN/END | result_dir=%s", result_dir)
    print(result_dir)
    print(pd.DataFrame([result]).to_string(index=False))


if __name__ == "__main__":
    main()
