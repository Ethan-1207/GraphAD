# -*- coding: utf-8 -*-
"""GraphAD 主实验（本文模型）。

用法（项目根目录）：
  python -m src.experiments.run_graphad                       # 完整模型
  python -m src.experiments.run_graphad --backbone gat --k 20
  python -m src.experiments.run_graphad --no-graph            # 消融：去人群图
  python -m src.experiments.run_graphad --no-imputer          # 消融：去模态补全
  python -m src.experiments.run_graphad --eval-pet-missing 0.5  # 缺失鲁棒性曲线
"""

import argparse

import torch

from src.config import ExperimentConfig, RESULT_DIR
from src.experiments.common import load_feature_matrix, prepare_experiment
from src.models.graphad import GraphAD
from src.training.evaluate import evaluate_full, format_metrics
from src.training.seed import set_seed
from src.training.trainer import predict, train_graphad
from src.utils import save_json


def run(cfg: ExperimentConfig, rebuild: bool = False, verbose: bool = False) -> dict:
    set_seed(cfg.seed)
    feat = load_feature_matrix(force_rebuild=rebuild)
    ctx = prepare_experiment(feat, seed=cfg.seed,
                             pet_missing_rate=cfg.eval_pet_missing_rate)

    n_mri = len(ctx["cols"]["mri"])
    dims = {"mri": n_mri, "pet": len(ctx["cols"]["pet"]),
            "cog": len(ctx["cols"]["cog"]), "cov": len(ctx["cols"]["cov"])}
    model = GraphAD(dims, emb_dim=cfg.emb_dim, backbone=cfg.backbone,
                    gnn_hidden=cfg.gnn_hidden, gnn_layers=cfg.gnn_layers,
                    dropout=cfg.dropout, use_graph=cfg.use_graph,
                    use_imputer=cfg.use_imputer)

    X_tr, obs_tr, y_tr = ctx["tensors"]["train"]
    X_va, obs_va, y_va = ctx["tensors"]["val"]
    # 转导式：把 train+val+test 拼在开发图上训练（节点索引见 ctx["dev_node_splits"]）
    X_dev = {m: torch.cat([X_tr[m], X_va[m],
                           ctx["tensors"]["test"][0][m]]) for m in dims}
    obs_dev = {m: torch.cat([obs_tr[m], obs_va[m],
                             ctx["tensors"]["test"][1][m]]) for m in dims}
    y_dev = torch.cat([y_tr, y_va, ctx["tensors"]["test"][2]])
    ei, ew = ctx["dev_graph"]
    node_splits = ctx["dev_node_splits"]

    model, history = train_graphad(
        model, X_dev, obs_dev, y_dev, node_splits,
        edge_index=ei if cfg.use_graph else None,
        edge_weight=ew if cfg.use_graph else None,
        lr=cfg.lr, weight_decay=cfg.weight_decay, max_epochs=cfg.max_epochs,
        patience=cfg.patience, recon_lambda=cfg.recon_lambda,
        train_missing_rate=cfg.train_missing_rate, verbose=verbose)

    out = {"config": cfg.to_dict(), "best_epoch": history[-1]["epoch"]}
    # 开发集内部测试
    X_te, obs_te, y_te = ctx["tensors"]["test"]
    idx_te = node_splits["test"]
    if cfg.use_graph:
        prob_dev = predict(model, X_dev, obs_dev, ei, ew)
        prob_te = prob_dev[idx_te.numpy()]
    else:
        prob_te = predict(model, X_te, obs_te)
    m = evaluate_full(y_te.numpy(), prob_te, seed=cfg.seed)
    out["test"] = m
    print(f"[GraphAD | test] {format_metrics(m)}")

    # 外部验证（ADNI-GO/2/3，独立图）
    X_ex, obs_ex, y_ex = ctx["tensors"]["external"]
    if len(y_ex) and len(set(y_ex.numpy())) > 1:
        ei_ex, ew_ex = ctx["ext_graph"]
        prob_ex = (predict(model, X_ex, obs_ex, ei_ex, ew_ex)
                   if cfg.use_graph else predict(model, X_ex, obs_ex))
        m_ex = evaluate_full(y_ex.numpy(), prob_ex, seed=cfg.seed)
        out["external"] = m_ex
        print(f"[GraphAD | external] {format_metrics(m_ex)}")

    out["history"] = history
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="gcn", choices=["gcn", "gat", "sage"])
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--similarity", default="mixed",
                    choices=["phenotype", "gene", "mixed"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--recon-lambda", type=float, default=0.1)
    ap.add_argument("--train-missing", type=float, default=0.3,
                    help="训练时随机遮 PET 比例")
    ap.add_argument("--eval-pet-missing", type=float, default=0.0,
                    help="评估模拟 PET 缺失率 0/0.2/0.5/1.0")
    ap.add_argument("--no-graph", action="store_true", help="消融：去人群图")
    ap.add_argument("--no-imputer", action="store_true", help="消融：去模态补全")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = ExperimentConfig(
        backbone=args.backbone, gnn_layers=args.layers, k=args.k,
        similarity=args.similarity, max_epochs=args.epochs, patience=args.patience,
        recon_lambda=args.recon_lambda, train_missing_rate=args.train_missing,
        eval_pet_missing_rate=args.eval_pet_missing,
        use_graph=not args.no_graph, use_imputer=not args.no_imputer,
        seed=args.seed,
    )
    result = run(cfg, rebuild=args.rebuild, verbose=args.verbose)

    tag = f"graphad_{cfg.backbone}_k{cfg.k}"
    if not cfg.use_graph:
        tag += "_nograph"
    if not cfg.use_imputer:
        tag += "_noimputer"
    if cfg.eval_pet_missing_rate > 0:
        tag += f"_miss{int(cfg.eval_pet_missing_rate * 100)}"
    save_json({k: v for k, v in result.items() if k != "history"},
              RESULT_DIR / f"{tag}_seed{cfg.seed}.json")
    print(f"\n结果已保存: {RESULT_DIR}")


if __name__ == "__main__":
    main()
