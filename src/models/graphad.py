# -*- coding: utf-8 -*-
"""GraphAD 全模型组装（论文四模块）：

  1) 模态编码层        —— MultiModalEncoder
  2) 模态掩码+补全(辅助) —— 训练随机遮 PET + CrossModalImputer（可关）
  3) 人群图消息传递(核心) —— GNNEncoder（可关：use_graph=False 退化为 MLP）
  4) 分类头            —— readout + MLP 输出 3 年内转 AD 概率

forward 返回 (logits, aux)，aux 含重建损失等，总损失由 trainer 组合：
  loss = CE(logits, y) + recon_lambda * recon_loss
"""

import torch
import torch.nn as nn

from src.config import GNN_BACKBONE, GNN_HIDDEN, GNN_LAYERS
from src.models.encoders import MultiModalEncoder
from src.models.gnn import GNNEncoder
from src.models.imputer import CrossModalImputer, recon_supervised_loss

MODALITIES = ["mri", "pet", "cog", "cov"]


class GraphAD(nn.Module):
    def __init__(self, dims: dict, emb_dim: int = 64,
                 backbone: str = GNN_BACKBONE, gnn_hidden: int = GNN_HIDDEN,
                 gnn_layers: int = GNN_LAYERS, dropout: float = 0.5,
                 use_graph: bool = True, use_imputer: bool = True):
        super().__init__()
        self.use_graph = use_graph
        self.use_imputer = use_imputer
        self.modalities = list(dims.keys())
        self.encoder = MultiModalEncoder(dims, emb_dim, dropout=0.3)
        if use_imputer:
            self.imputer = CrossModalImputer(emb_dim)
        n_mlp_in = emb_dim * len(self.modalities) + len(self.modalities)  # 拼接+缺失指示
        if use_graph:
            self.gnn = GNNEncoder(n_mlp_in, gnn_hidden, emb_dim,
                                  backbone, gnn_layers, dropout)
            head_in = emb_dim
        else:
            head_in = n_mlp_in
        self.head = nn.Sequential(
            nn.Linear(head_in, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, features: dict, observed: dict, edge_index=None,
                edge_weight=None, train_missing_rate: float = 0.0):
        # 先编码一份"干净"嵌入（用于重建监督的真值）
        clean_emb, groundtruth = self.encoder(features, observed, train_missing_rate=0.0)
        # 再按训练掩码编码（随机遮 PET）
        masked_emb, obs_after = self.encoder(features, observed,
                                             train_missing_rate=train_missing_rate)
        recon_loss = torch.tensor(0.0, device=next(iter(features.values())).device)
        if self.use_imputer and self.training:
            recon_emb = self.imputer(masked_emb, obs_after)
            recon_loss = recon_supervised_loss(recon_emb, clean_emb, obs_after, groundtruth)
        elif self.use_imputer and not self.training:
            recon_emb = self.imputer(masked_emb, obs_after)
        else:
            recon_emb = None

        # 组装节点表示：观测处用（掩码后）嵌入，缺失处用重建嵌入补位
        zs, miss_inds = [], []
        for m in self.modalities:
            obs = obs_after[m]
            emb = masked_emb[m]
            if recon_emb is not None:
                emb = torch.where(obs.unsqueeze(1), emb, recon_emb[m])
            zs.append(emb)
            miss_inds.append((~obs).float().unsqueeze(1))
        z = torch.cat(zs + miss_inds, dim=1)  # [N, M*D + M]

        if self.use_graph:
            if edge_index is None:
                raise ValueError("use_graph=True 需要 edge_index（人群图）")
            z = self.gnn(z, edge_index, edge_weight)
        logits = self.head(z)
        return logits, {"recon_loss": recon_loss}
