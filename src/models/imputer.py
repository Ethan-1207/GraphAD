# -*- coding: utf-8 -*-
"""模态补全模块（辅助手段，非卖点）：跨模态注意力重建缺失模态嵌入。

思路：缺检查时，用该病人自己的其他检查把缺的部分"猜"出来。
补得准不准由重建损失把关（只对"真实存在但被遮蔽"的位置计算 MSE）。
"""

import torch
import torch.nn as nn


class CrossModalImputer(nn.Module):
    """对每个模态，用其余模态嵌入做 attention 加权组合后重建。"""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.attn = nn.Parameter(torch.ones(1) * 0.5)  # 可学习的注意力温度缩放

    def forward(self, embeddings: dict, observed: dict):
        """返回 recon_embeddings: dict {模态名 -> Tensor[N, D]}。

        对每个模态 m，用其余模态的嵌入做注意力加权组合，得到 m 的重建嵌入；
        重建损失单独用 recon_supervised_loss 计算（只监督"真实存在但被遮蔽"
        的位置，保证补全器不靠"瞎猜永远缺的模态"刷分）。
        """
        mods = list(embeddings.keys())
        recon_emb = {}
        for m in mods:
            others = [o for o in mods if o != m]
            keys = torch.stack([embeddings[o] for o in others], dim=1)  # [N, M-1, D]
            mask_o = torch.stack([observed[o].float() for o in others], dim=1)  # [N, M-1]
            # 简化注意力：query=待重建模态当前嵌入, key=其余模态嵌入
            q = embeddings[m].unsqueeze(1)  # [N,1,D]
            score = (q * keys).sum(-1) * self.attn  # [N, M-1]
            score = score.masked_fill(mask_o == 0, -1e9)
            attn_w = torch.softmax(score, dim=1) * mask_o  # [N, M-1]
            recon_emb[m] = (attn_w.unsqueeze(-1) * keys).sum(1)  # [N, D]
        return recon_emb


def recon_supervised_loss(recon_emb: dict, clean_emb: dict,
                          observed_after_mask: dict, groundtruth: dict):
    """对"真实存在但被遮蔽"的位置计算 MSE 重建损失（论文的重建损失项）。"""
    loss, n = 0.0, 0
    for m in recon_emb:
        should_sup = groundtruth[m] & ~observed_after_mask[m]  # 有真值但被遮
        if should_sup.any():
            se = ((recon_emb[m] - clean_emb[m]) ** 2).mean(dim=1)
            loss = loss + se[should_sup].sum()
            n += int(should_sup.sum())
    if n == 0:
        return torch.tensor(0.0, device=next(iter(recon_emb.values())).device)
    return loss / n
