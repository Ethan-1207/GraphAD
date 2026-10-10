# -*- coding: utf-8 -*-
"""消融实验（对应飞书"消融实验"设计，一键跑完全部消融表）：

  1) 去人群图 / 去模态补全（第 1 步优先验证，决定论文故事）
  2) 图结构：GCN vs GAT vs GraphSAGE；KNN k ∈ {5,10,20}；相似度度量
  3) 鲁棒性曲线：评估时 PET 缺失率 0/20/50/100%
  4) 补全质量：重建损失权重 recon_lambda = 0 vs 默认

用法：
  python -m src.experiments.run_ablation               # 全量（较久）
  python -m src.experiments.run_ablation --quick       # 短 epoch 快速验证管线
"""

import argparse

import pandas as pd

from src.config import ExperimentConfig, RESULT_DIR
from src.experiments.run_graphad import run as run_graphad_cfg
from src.training.seed import set_seed
from src.utils import ensure_dir

# 消融设置名 → ExperimentConfig 覆盖项
ABLATIONS = {
    "full":                {},
    "w/o_imputer":         {"use_imputer": False},
    "w/o_graph":           {"use_graph": False},
    "gat":                 {"backbone": "gat"},
    "sage":                {"backbone": "sage"},
    "k5":                  {"k": 5},
    "k20":                 {"k": 20},
    "sim_phenotype":       {"similarity": "phenotype"},
    "sim_gene":            {"similarity": "gene"},
    "recon_lambda0":       {"recon_lambda": 0.0},
    "pet_miss20":          {"eval_pet_missing_rate": 0.2},
    "pet_miss50":          {"eval_pet_missing_rate": 0.5},
    "pet_miss100":         {"eval_pet_missing_rate": 1.0},
}

QUICK_EPOCHS = 40


def flatten_result(name: str, r: dict) -> dict:
    row = {"setting": name, **{k: v for k, v in r["config"].items()}}
    for split in ("test", "external"):
        m = r.get(split)
        if not m:
            continue
        for k in ("ACC", "AUC", "Sensitivity", "Specificity", "F1", "n"):
            row[f"{split}_{k}"] = m.get(k)
        ci = m.get("CI95", {}).get("AUC")
        if ci:
            row[f"{split}_AUC_CI95"] = f"[{ci[0]:.3f},{ci[1]:.3f}]"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="短 epoch 冒烟测试")
    ap.add_argument("--only", nargs="*", default=None,
                    help="只跑指定消融设置，如 --only full w/o_graph")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    settings = args.only if args.only else list(ABLATIONS)
    rows = []
    for name in settings:
        if name not in ABLATIONS:
            print(f"跳过未知设置: {name}")
            continue
        cfg_kwargs = dict(ABLATIONS[name])
        if args.quick:
            cfg_kwargs.setdefault("max_epochs", QUICK_EPOCHS)
            cfg_kwargs.setdefault("patience", QUICK_EPOCHS)
        print(f"\n=== 消融: {name} ===")
        r = run_graphad_cfg(ExperimentConfig(name=name, seed=args.seed, **cfg_kwargs))
        rows.append(flatten_result(name, r))

    ensure_dir(RESULT_DIR)
    out_csv = RESULT_DIR / f"ablation_seed{args.seed}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n消融表已保存: {out_csv}")


if __name__ == "__main__":
    main()
