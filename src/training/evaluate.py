# -*- coding: utf-8 -*-
"""评估：ACC / AUC / 敏感性 / 特异性 / F1 + bootstrap 95% CI。"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """y_prob: 转化(positive=1)的概率。"""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {
        "ACC": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "Sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "Specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "n": len(y_true),
    }


def bootstrap_ci(y_true, y_prob, metric_fn=binary_metrics,
                 n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> dict:
    """对每个指标做 bootstrap，返回 {指标: (低, 高)}。"""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    keys = [k for k in binary_metrics(y_true, y_prob) if k != "n"]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y_true[idx])) < 2:
            continue
        m = metric_fn(y_true[idx], y_prob[idx])
        for k in keys:
            samples[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 100 * alpha / 2)),
                float(np.nanpercentile(v, 100 * (1 - alpha / 2))))
            for k, v in samples.items() if v}


def evaluate_full(y_true, y_prob, with_ci: bool = True, seed: int = 42) -> dict:
    m = binary_metrics(y_true, y_prob)
    if with_ci:
        ci = bootstrap_ci(y_true, y_prob, seed=seed)
        m["CI95"] = ci
    return m


def format_metrics(m: dict) -> str:
    base = (f"ACC={m['ACC']:.3f} AUC={m['AUC']:.3f} "
            f"Sen={m['Sensitivity']:.3f} Spe={m['Specificity']:.3f} "
            f"F1={m['F1']:.3f} (n={m['n']})")
    ci = m.get("CI95", {}).get("AUC")
    if ci:
        base += f"  AUC-95%CI=[{ci[0]:.3f},{ci[1]:.3f}]"
    return base
