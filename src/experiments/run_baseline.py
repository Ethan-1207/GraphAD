# -*- coding: utf-8 -*-
"""传统基线实验：LR / RF / SVM（临床变量 + 影像特征）。

用法（项目根目录）：
  python -m src.experiments.run_baseline --model lr
  python -m src.experiments.run_baseline --model all
"""

import argparse

import pandas as pd

from src.experiments.common import load_feature_matrix, modality_columns
from src.models.baselines import BASELINE_NAMES, make_baseline
from src.training.evaluate import evaluate_full, format_metrics
from src.training.seed import set_seed
from src.config import RESULT_DIR
from src.utils import ensure_dir, save_json


def run(model_name: str, seed: int = 42, rebuild: bool = False) -> dict:
    set_seed(seed)
    feat = load_feature_matrix(force_rebuild=rebuild)
    cols = modality_columns(feat)
    feat_cols = [c for cc in cols.values() for c in cc]

    from src.data.split import make_splits
    splits = make_splits(feat, seed=seed)
    X_tr = splits["train"][feat_cols]
    y_tr = splits["train"]["label"]

    out = {"model": model_name, "seed": seed}
    for split_name in ["test", "external"]:
        X_te = splits[split_name][feat_cols]
        y_te = splits[split_name]["label"]
        if len(set(y_te)) < 2:
            print(f"[{split_name}] 只有一个类别，跳过")
            continue
        pipe = make_baseline(model_name, seed)
        pipe.fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_te)[:, 1]
        m = evaluate_full(y_te, prob, seed=seed)
        out[split_name] = m
        print(f"[{model_name.upper()} | {split_name}] {format_metrics(m)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=BASELINE_NAMES + ["all"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rebuild", action="store_true", help="强制重建特征矩阵缓存")
    args = ap.parse_args()

    names = BASELINE_NAMES if args.model == "all" else [args.model]
    results = [run(n, seed=args.seed, rebuild=args.rebuild) for n in names]

    ensure_dir(RESULT_DIR)
    pd.DataFrame([{**{"model": r["model"], "seed": r["seed"]},
                   **{f"{s}_{k}": v for s in ("test", "external") if s in r
                      for k, v in r[s].items() if k != "CI95"}}
                  for r in results]).to_csv(RESULT_DIR / "baselines.csv", index=False)
    save_json(results, RESULT_DIR / "baselines_full.json")
    print(f"\n结果已保存: {RESULT_DIR / 'baselines.csv'}")


if __name__ == "__main__":
    main()
