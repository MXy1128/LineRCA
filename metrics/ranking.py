from __future__ import annotations

"""根因实体排序的统一指标实现，所有模型必须调用这里以保证公平。"""
import numpy as np
import pandas as pd

DEFAULT_KS = (1, 2, 3)
METRIC_COLUMNS = ([f"Hit@{k}" for k in DEFAULT_KS] +
                  [f"Precision@{k}" for k in DEFAULT_KS] +
                  [f"Recall@{k}" for k in DEFAULT_KS] +
                  [f"MAP@{k}" for k in DEFAULT_KS] +
                  [f"NDCG@{k}" for k in DEFAULT_KS] + ["MRR"])


def normalize_by_scenario(df: pd.DataFrame, values: pd.Series | np.ndarray) -> pd.Series:
    tmp=df[["scenario_id"]].copy();tmp["_v"]=pd.to_numeric(values,errors="coerce").fillna(0.0)
    lo=tmp["_v"].groupby(tmp["scenario_id"]).transform("min");hi=tmp["_v"].groupby(tmp["scenario_id"]).transform("max")
    return ((tmp["_v"]-lo)/(hi-lo+1e-9)).fillna(0.0)


def ranking_metrics_arrays(y_true: np.ndarray, scores: np.ndarray, ks=DEFAULT_KS) -> dict[str,float]:
    """评价多标签根因排序。

    论文沿用的 PR@K 实际含义是 Hit@K：前 K 个候选中只要命中一个真实根因即为 1。
    MAP 考虑所有真实根因的位置，MRR 只看第一个真实根因的倒数排名，NDCG 同时
    考虑相关性和排名折损。无根因的正常样本不参与定位指标。
    """
    y=np.asarray(y_true);s=np.asarray(scores)
    if y.shape!=s.shape or y.ndim!=2:raise ValueError("y_true and scores must be equal 2-D arrays")
    if any(k<1 or k>y.shape[1] for k in ks):raise ValueError(f"K must be within [1,{y.shape[1]}]")
    values={name:{k:[] for k in ks} for name in ("Hit","Precision","Recall","MAP","NDCG")};rr=[]
    for truth,score in zip(y,s):
        total=int(truth.sum())
        if total<=0:continue
        # rel 是按预测分数降序排列后的 0/1 相关性序列。
        rel=truth[np.argsort(-score)].astype(int)
        first=np.flatnonzero(rel);rr.append(1.0/(first[0]+1) if len(first) else 0.0)
        for k in ks:
            top=rel[:k];hits=int(top.sum());values["Hit"][k].append(float(hits>0));values["Precision"][k].append(hits/k);values["Recall"][k].append(hits/total)
            running=np.cumsum(top);ap=float(np.sum((running/np.arange(1,k+1))*top)/min(total,k));values["MAP"][k].append(ap)
            discounts=1/np.log2(np.arange(2,k+2));dcg=float(np.sum(top*discounts));ideal=float(np.sum(discounts[:min(total,k)]));values["NDCG"][k].append(dcg/ideal if ideal else 0.0)
    out={}
    for name in ("Hit","Precision","Recall","MAP","NDCG"):
        out.update({f"{name}@{k}":float(np.mean(values[name][k])) if values[name][k] else 0.0 for k in ks})
    out["MRR"]=float(np.mean(rr)) if rr else 0.0
    return out


def localization_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """按论文表格约定输出 PR、MAP、MRR、NDCG 的完整 K 取值。"""
    metrics = ranking_metrics_arrays(y_true, scores, (1, 3, 4, 5, 7, 10))
    output = {f"PR@{k}": metrics[f"Hit@{k}"] for k in (1, 3, 5, 7, 10)}
    output.update({f"MAP@{k}": metrics[f"MAP@{k}"] for k in (3, 4, 5, 7, 10)})
    output["MRR"] = metrics["MRR"]
    output.update({f"NDCG@{k}": metrics[f"NDCG@{k}"] for k in (1, 3, 5, 7, 10)})
    return output


def ranking_metrics(df: pd.DataFrame, score_col: str, ks=DEFAULT_KS) -> dict[str,float]:
    groups=[]
    for _,g in df.groupby("scenario_id",sort=False):
        groups.append((g["is_true_root"].to_numpy(int),pd.to_numeric(g[score_col],errors="coerce").fillna(0).to_numpy(float)))
    if not groups:return {k:0.0 for k in METRIC_COLUMNS}
    n={len(y) for y,_ in groups}
    if len(n)!=1:raise ValueError("every scenario must contain the same number of candidates")
    return ranking_metrics_arrays(np.stack([y for y,_ in groups]),np.stack([s for _,s in groups]),ks)


def percent_table(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy();skip={"Dataset","Experiment","Model","Ablation","split","seed","best_epoch","params"}
    for col in out.columns:
        if col not in skip:out[col]=out[col].map(lambda x:f"{float(x)*100:.2f}%" if pd.notna(x) else "")
    return out


def format_metric_line(metrics: dict[str,float]) -> str:
    return " | ".join(f"{key}={metrics[key]:.6f}" for key in METRIC_COLUMNS if key in metrics)


def markdown_table(df: pd.DataFrame) -> str:
    cols=df.columns.tolist();rows=["| "+" | ".join(cols)+" |","| "+" | ".join(["---"]*len(cols))+" |"]
    rows += ["| "+" | ".join(str(row[c]) for c in cols)+" |" for _,row in df.iterrows()]
    return "\n".join(rows)
