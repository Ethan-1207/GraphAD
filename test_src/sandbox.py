# -*- coding: utf-8 -*-
"""同学实验沙盒：一行拿到数据 / 图 / 评估，专心试自己的方法。

约定（重要）：
  - 不要改 src/ 里的任何文件；你的实验全部写在本目录；
  - 数据划分、评估指标、缺失场景模拟一律复用沙盒提供的函数，
    保证你的结果和主实验可比；
  - 结果存到 test_result/，不要覆盖 result/ 里主实验的结果。

快速上手见同目录 README.md 和 my_experiments.py（模板）。
"""

import sys
from pathlib import Path

# 保证从任何目录运行都能 import src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.config import ExperimentConfig  # noqa: E402
from src.experiments.common import load_feature_matrix, prepare_experiment  # noqa: E402
from src.models.graphad import GraphAD  # noqa: E402
from src.training.evaluate import evaluate_full, format_metrics  # noqa: E402
from src.training.seed import set_seed  # noqa: E402
from src.training.trainer import predict, train_graphad  # noqa: E402

__all__ = [
    "load_data", "build_model", "train", "predict", "evaluate",
    "format_metrics", "set_seed", "ExperimentConfig",
    "DEVICE",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(seed: int = 42, pet_missing_rate: float = 0.0, rebuild: bool = False):
    """一键拿到全部实验材料（首次运行会先构建特征矩阵，约 1-2 分钟）。

    返回 dict，常用字段：
      ctx["tensors"]["train"|"val"|"test"|"external"]  → (features, observed, y)
      ctx["dev_graph"] / ctx["ext_graph"]              → (edge_index, edge_weight)
      ctx["dev_node_splits"]                           → 图上 train/val/test 节点索引
      ctx["cols"]                                      → 各模态特征列名
    """
    feat = load_feature_matrix(force_rebuild=rebuild)
    return prepare_experiment(feat, seed=seed, pet_missing_rate=pet_missing_rate)


def build_model(ctx, **overrides) -> GraphAD:
    """按 ctx 自动推断维度构建 GraphAD（换自己的模型时仅作参考）。"""
    cfg = ExperimentConfig(**overrides)
    dims = {m: len(cols) for m, cols in ctx["cols"].items()}
    return GraphAD(dims, emb_dim=cfg.emb_dim, backbone=cfg.backbone,
                   gnn_hidden=cfg.gnn_hidden, gnn_layers=cfg.gnn_layers,
                   dropout=cfg.dropout, use_graph=cfg.use_graph,
                   use_imputer=cfg.use_imputer)


def train(model, ctx, cfg: ExperimentConfig | None = None, verbose: bool = False):
    """用统一协议训练（val AUC 早停），返回 (model, history)。"""
    cfg = cfg or ExperimentConfig()
    X_tr, obs_tr, y_tr = ctx["tensors"]["train"]
    X_va, obs_va, _ = ctx["tensors"]["val"]
    X_te, obs_te, _ = ctx["tensors"]["test"]
    mods = list(X_tr.keys())
    X_dev = {m: torch.cat([X_tr[m], X_va[m], X_te[m]]) for m in mods}
    obs_dev = {m: torch.cat([obs_tr[m], obs_va[m], obs_te[m]]) for m in mods}
    y_dev = torch.cat([y_tr, ctx["tensors"]["val"][2], ctx["tensors"]["test"][2]])
    ei, ew = ctx["dev_graph"]
    return train_graphad(
        model, X_dev, obs_dev, y_dev, ctx["dev_node_splits"],
        edge_index=ei if model.use_graph else None,
        edge_weight=ew if model.use_graph else None,
        lr=cfg.lr, weight_decay=cfg.weight_decay, max_epochs=cfg.max_epochs,
        patience=cfg.patience, recon_lambda=cfg.recon_lambda,
        train_missing_rate=cfg.train_missing_rate, verbose=verbose)


def evaluate(model, ctx, split: str = "test", with_ci: bool = True, seed: int = 42) -> dict:
    """统一评估：test/val 在开发图上前向后取本子集节点；external 用外部图。"""
    X, obs, y = ctx["tensors"][split]
    if model.use_graph:
        if split == "external":
            ei, ew = ctx["ext_graph"]
            prob = predict(model, X, obs, ei, ew)
        else:
            # 转导式：必须整张开发图前向（图结构引用全部节点），再切本子集
            mods = list(X.keys())
            names = ["train", "val", "test"]
            Xs = [ctx["tensors"][n][0] for n in names]
            obs_all = {m: torch.cat([ctx["tensors"][n][1][m] for n in names]) for m in mods}
            ei, ew = ctx["dev_graph"]
            prob = predict(model, {m: torch.cat([t[m] for t in Xs]) for m in mods},
                           obs_all, ei, ew)
            offset = {"train": 0, "val": len(ctx["tensors"]["train"][2]),
                      "test": len(ctx["tensors"]["train"][2]) + len(ctx["tensors"]["val"][2])}
            start = offset[split]
            prob = prob[start:start + len(y)]
    else:
        prob = predict(model, X, obs)
    return evaluate_full(y.numpy(), prob, with_ci=with_ci, seed=seed)
