# -*- coding: utf-8 -*-
"""模态编码层 + 模态掩码。

每个受试者的三模态原始特征（MRI 皮质厚度/海马、PET SUVR、认知量表）
分别经全连接层映射到统一嵌入空间；缺失模态用可学习 mask token 填充。
"""

import torch
import torch.nn as nn

from src.config import COG_FEATURES, COVARIATES, EMB_DIM, PET_FEATURES


class ModalityEncoder(nn.Module):
    """单模态 FC 编码器：Linear → BN → ReLU → Dropout。"""

    def __init__(self, in_dim: int, emb_dim: int = EMB_DIM, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiModalEncoder(nn.Module):
    """三模态编码 + 缺失掩码。

    输入:
      features: dict {模态名 -> Tensor[N, d]}（缺失位置可用任意值占位）
      observed: dict {模态名 -> BoolTensor[N]}，True=该模态真实存在
      train_missing_rate: 训练时额外随机遮蔽已存在 PET 的比例（模拟缺失）
    返回:
      embeddings: dict {模态名 -> Tensor[N, emb_dim]}（缺失处为 mask token）
      observed_dict: 合并"真实缺失+随机遮蔽"后的观测掩码（供 imputer / 判别用）
    """

    def __init__(self, dims: dict, emb_dim: int = EMB_DIM, dropout: float = 0.3):
        super().__init__()
        self.modalities = list(dims.keys())
        self.encoders = nn.ModuleDict(
            {m: ModalityEncoder(d, emb_dim, dropout) for m, d in dims.items()})
        self.mask_tokens = nn.ParameterDict(
            {m: nn.Parameter(torch.randn(emb_dim) * 0.02) for m in self.modalities})

    def forward(self, features: dict, observed: dict,
                train_missing_rate: float = 0.0, pet_modality: str = "pet"):
        embeddings, observed_out = {}, {}
        for m in self.modalities:
            obs = observed[m].clone()
            # 训练时对 PET 随机遮蔽（模态掩码策略，模拟真实缺失先验）
            if self.training and m == pet_modality and train_missing_rate > 0:
                drop = torch.rand_like(obs.float()) < train_missing_rate
                obs = obs & ~drop
            x = features[m].clone()
            x[~obs] = 0.0  # 缺失位置清零再编码
            emb = self.encoders[m](x)
            token = self.mask_tokens[m].unsqueeze(0).expand(emb.size(0), -1)
            embeddings[m] = torch.where(obs.unsqueeze(1), emb, token)
            observed_out[m] = obs
        return embeddings, observed_out


def default_dims(n_mri: int):
    """按 config 默认特征名返回各模态维度（除 MRI 外均已知）。"""
    return {"mri": n_mri, "pet": len(PET_FEATURES), "cog": len(COG_FEATURES),
            "cov": len(COVARIATES)}
