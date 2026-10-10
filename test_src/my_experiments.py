# -*- coding: utf-8 -*-
"""同学实验模板：把你的新方法填进 MyModel，跑法见文件底部 main()。

原则：
  1) 只改这个文件（或复制多份），不动 src/；
  2) 训练 / 评估用 sandbox 提供的统一协议，保证公平可比；
  3) 结果存 test_result/（沙盒已自动指向），主实验的 result/ 不要碰。

快速跑：
  python -m test_src.my_experiments                # 模板示例：只卷积认知模态的 GAT
  python -m test_src.my_experiments --quick        # 短 epoch 冒烟
"""

import argparse

import torch
import torch.nn as nn

import test_src.sandbox as sb
from src.config import ExperimentConfig, TEST_RESULT_DIR
from src.utils import save_json


class MyModel(nn.Module):
    """示例：在 GraphAD 基础上换一个自己的设计。

    这个示例展示三种常见改法：
      a) 换 GNN 层 / 聚合方式（这里用 GAT 演示）；
      b) 只挑部分模态；
      c) 加自定义辅助损失（self.aux_loss）。
    你可以直接把 forward 改成任何结构，只要：
      forward(features, observed, edge_index, edge_weight, train_missing_rate)
      → (logits [N,2], aux: {"recon_loss": tensor})
    """

    def __init__(self, dims: dict, emb_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        self.use_graph = True  # sandbox 依赖此标记决定是否传图
        from src.models.encoders import MultiModalEncoder
        from src.models.gnn import GNNEncoder
        from src.models.imputer import CrossModalImputer

        self.encoder = MultiModalEncoder(dims, emb_dim)
        self.imputer = CrossModalImputer(emb_dim)
        self.gnn = GNNEncoder(emb_dim * len(dims) + len(dims), emb_dim, emb_dim,
                              backbone="gat", num_layers=2, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 2))
        self.aux_loss = torch.tensor(0.0)

    def forward(self, features, observed, edge_index=None, edge_weight=None,
                train_missing_rate: float = 0.0):
        emb, obs = self.encoder(features, observed,
                                train_missing_rate=train_missing_rate)
        recon = self.imputer(emb, obs)
        zs = [torch.where(obs[m].unsqueeze(1), emb[m], recon[m])
              for m in emb]
        z = torch.cat(zs + [(~obs[m]).float().unsqueeze(1) for m in emb], dim=1)
        z = self.gnn(z, edge_index, edge_weight)
        logits = self.head(z)
        self.aux_loss = torch.tensor(0.0, device=logits.device)
        return logits, {"recon_loss": self.aux_loss}


def run(seed: int = 42, quick: bool = False, verbose: bool = False):
    sb.set_seed(seed)
    print(f"设备: {sb.DEVICE}")
    ctx = sb.load_data(seed=seed)

    dims = {m: len(cols) for m, cols in ctx["cols"].items()}
    model = MyModel(dims).to(sb.DEVICE)

    cfg = ExperimentConfig(max_epochs=40 if quick else 300,
                           patience=40 if quick else 30)
    model, history = sb.train(model, ctx, cfg, verbose=verbose)

    for split in ["test", "external"]:
        m = sb.evaluate(model, ctx, split=split, seed=seed)
        print(f"[MyModel | {split}] {sb.format_metrics(m)}")

    # ---- 结果落盘（test_result/）----
    best = history[-1]["epoch"]
    out = {"model": "MyModel", "seed": seed, "best_epoch": best,
           "test": {k: v for k, v in sb.evaluate(model, ctx, "test", seed=seed).items()},
           "external": {k: v for k, v in sb.evaluate(model, ctx, "external", seed=seed).items()}}
    save_json(out, TEST_RESULT_DIR / f"my_model_seed{seed}.json")
    print(f"\n结果已保存: {TEST_RESULT_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run(seed=args.seed, quick=args.quick, verbose=args.verbose)


if __name__ == "__main__":
    main()
