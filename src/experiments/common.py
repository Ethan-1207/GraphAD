# -*- coding: utf-8 -*-
"""实验公共层：特征矩阵缓存加载、张量化、评估缺失场景模拟。"""

import numpy as np
import pandas as pd
import torch

from src.config import PROCESSED_DIR, COG_FEATURES, COVARIATES, PET_FEATURES
from src.data.features import build_feature_matrix
from src.data.split import make_splits
from src.utils import ensure_dir

MRI_PREFIX = "mri_"
MODALITY_COLS = {
    "mri": None,  # 运行时按前缀发现
    "pet": PET_FEATURES,
    "cog": COG_FEATURES,
    "cov": COVARIATES,
}


def load_feature_matrix(force_rebuild: bool = False) -> pd.DataFrame:
    """优先读缓存 data/processed/feature_matrix.csv，否则重建并缓存。"""
    cache = PROCESSED_DIR / "feature_matrix.csv"
    if cache.exists() and not force_rebuild:
        return pd.read_csv(cache)
    return build_feature_matrix(save=True)


def modality_columns(feat: pd.DataFrame) -> dict:
    cols = dict(MODALITY_COLS)
    cols["mri"] = [c for c in feat.columns if c.startswith(MRI_PREFIX)]
    return cols


def fit_scaler(train_df: pd.DataFrame, cols: list) -> dict:
    st = {}
    for c in cols:
        v = train_df[c].astype(float)
        st[c] = (float(v.mean()), float(v.std() + 1e-8))
    return st


def simulate_pet_missing(df: pd.DataFrame, rate: float, seed: int = 42) -> pd.Series:
    """在"真实有 PET"的人里按 rate 决定性地遮蔽（固定 rng，评估可复现）。

    返回 BoolTensor：True = 评估时该 PET 视为观测。
    """
    has_pet = df["pet_fdg_suvr"].notna() | df["pet_av45_suvr"].notna()
    rng = np.random.default_rng(seed)
    keep = pd.Series(True, index=df.index)
    idx = df.index[has_pet]
    drop = rng.random(len(idx)) < rate
    keep.loc[idx[drop]] = False
    return keep


def to_tensors(feat: pd.DataFrame, scaler: dict, cols_by_mod: dict,
               pet_keep: pd.Series | None = None, device: str = "cpu"):
    """DataFrame → (features dict, observed dict, labels LongTensor)。

    observed: 模态真实可用性（PET 可叠加评估模拟缺失 pet_keep）。
    """
    features, observed = {}, {}
    for m, cols in cols_by_mod.items():
        X = np.zeros((len(feat), len(cols)), dtype=np.float32)
        obs = np.zeros(len(feat), dtype=bool)
        for j, c in enumerate(cols):
            v = feat[c].astype(float)
            mu, sd = scaler[c]
            obs |= v.notna().values
            X[:, j] = ((v - mu) / sd).fillna(0.0).values
        features[m] = torch.tensor(X, device=device)
        observed[m] = torch.tensor(obs, device=device)
    if pet_keep is not None:
        observed["pet"] = observed["pet"] & torch.tensor(pet_keep.values, device=device)
    y = torch.tensor(feat["label"].astype(int).values, dtype=torch.long, device=device)
    return features, observed, y


def prepare_experiment(feat: pd.DataFrame, seed: int = 42, pet_missing_rate: float = 0.0):
    """一键准备：划分 → 标准化 → 张量 → 图。返回 dict。

    评估侧（test/external）按 pet_missing_rate 模拟 PET 缺失场景
    （0/0.2/0.5/1.0 对应论文鲁棒性曲线）。
    """
    from src.graph.build_graph import build_knn_graph

    splits = make_splits(feat, seed=seed)
    cols = modality_columns(feat)
    all_cols = [c for cc in cols.values() for c in cc]
    scaler = fit_scaler(splits["train"], all_cols)

    tensors, rids = {}, {}
    for name in ["train", "val", "test", "external"]:
        df = splits[name]
        pet_keep = None
        if name in ("test", "external") and pet_missing_rate > 0:
            pet_keep = simulate_pet_missing(df, pet_missing_rate, seed=seed)
        X, obs, y = to_tensors(df, scaler, cols, pet_keep=pet_keep)
        tensors[name] = (X, obs, y)
        rids[name] = df["RID"].values

    # 转导式：开发集（train+val+test）一张图；外部队列单独一张图
    dev = pd.concat([splits["train"], splits["val"], splits["test"]], ignore_index=True)
    edge_index_dev, edge_weight_dev = build_knn_graph(dev, similarity="mixed")
    n_tr = len(splits["train"])
    n_va = len(splits["val"])
    node_splits = {"train": torch.arange(0, n_tr),
                   "val": torch.arange(n_tr, n_tr + n_va),
                   "test": torch.arange(n_tr + n_va, len(dev))}
    ext_edge_index, ext_edge_weight = build_knn_graph(splits["external"], similarity="mixed")

    return {"splits": splits, "scaler": scaler, "cols": cols,
            "tensors": tensors, "rids": rids,
            "dev_graph": (edge_index_dev, edge_weight_dev),
            "dev_node_splits": node_splits,
            "ext_graph": (ext_edge_index, ext_edge_weight)}
