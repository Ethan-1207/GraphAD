# -*- coding: utf-8 -*-
"""训练循环（全批次，适配人群图的小样本规模）。

损失 = 交叉熵 + recon_lambda * 模态重建损失（论文四模块的损失设计）。
早停准则：验证集 AUC。
"""

import numpy as np
import torch
import torch.nn as nn

from src.training.evaluate import binary_metrics

GraphADLike = "GraphAD"  # 类型提示占位（字符串注解，避免循环导入）


def train_graphad(model: "GraphADLike", features: dict, observed: dict, labels: torch.Tensor,
                  node_splits: dict, edge_index=None, edge_weight=None,
                  lr: float = 1e-3, weight_decay: float = 1e-4,
                  max_epochs: int = 300, patience: int = 30,
                  recon_lambda: float = 0.1, train_missing_rate: float = 0.3,
                  verbose: bool = False):
    """node_splits: {"train": LongTensor 节点索引, "val": ...}。

    全图前向（转导式），仅对 train 节点回传损失，val AUC 早停。
    返回 (model, history)。
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    tr_idx, va_idx = node_splits["train"], node_splits["val"]
    y = labels
    best_auc, best_state, bad = -1.0, None, 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        logits, aux = model(features, observed, edge_index, edge_weight,
                            train_missing_rate=train_missing_rate)
        loss = ce(logits[tr_idx], y[tr_idx]) + recon_lambda * aux["recon_loss"]
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits_eval, _ = model(features, observed, edge_index, edge_weight)
            prob = torch.softmax(logits_eval, dim=1)[:, 1].numpy()
        y_np = y.numpy()
        val_m = binary_metrics(y_np[va_idx.numpy()], prob[va_idx.numpy()])
        history.append({"epoch": epoch, "loss": float(loss), "val_AUC": val_m["AUC"]})

        if val_m["AUC"] > best_auc:
            best_auc, bad = val_m["AUC"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose and epoch % 20 == 0:
            print(f"epoch {epoch:3d} loss={loss:.4f} val_AUC={val_m['AUC']:.3f}")
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict(model, features, observed, edge_index=None, edge_weight=None):
    """返回全图节点的转化概率。"""
    model.eval()
    with torch.no_grad():
        logits, _ = model(features, observed, edge_index, edge_weight)
        return torch.softmax(logits, dim=1)[:, 1].numpy()
