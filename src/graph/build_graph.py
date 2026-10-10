# -*- coding: utf-8 -*-
"""人群图构建（核心卖点）：病人为节点，表型/基因相似度 KNN 连边。

说明（Parisot 2017 转导式设定）：图的连边只依赖特征（认知量表 + APOE），
不使用标签，因此不存在标签泄漏；对开发集一次性建图、train/val/test 作为
图上的不同节点子集训练/评估，与 Parisot 人群图框架一致。
"""

import torch
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from src.config import GRAPH_GENE_COLS, GRAPH_K, GRAPH_MIXED_ALPHA, GRAPH_PHENO_COLS


def _z(df, cols):
    X = df[cols].astype(float).copy()
    X = X.fillna(X.median())
    return StandardScaler().fit_transform(X)


def similarity_matrix(feat_df, similarity: str = "mixed", alpha: float = GRAPH_MIXED_ALPHA):
    """返回 [N,N] 相似度矩阵（越大越相似）。"""
    P = _z(feat_df, GRAPH_PHENO_COLS)
    G = _z(feat_df, GRAPH_GENE_COLS)
    # 负欧氏距离即相似度
    import numpy as np
    sp = -((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    sg = -((G[:, None, :] - G[None, :, :]) ** 2).sum(-1)
    if similarity == "phenotype":
        return sp
    if similarity == "gene":
        return sg
    if similarity == "mixed":
        return alpha * sp + (1 - alpha) * sg
    raise ValueError(f"未知相似度: {similarity}")


def build_knn_graph(feat_df, k: int = GRAPH_K, similarity: str = "mixed",
                    alpha: float = GRAPH_MIXED_ALPHA):
    """构建 KNN 图。返回 (edge_index LongTensor[2,E], edge_weight FloatTensor[E])。

    对称化（互为近邻合并），边权按目标节点的邻居做 softmax 归一，
    避免枢纽节点边权爆炸。
    """
    import numpy as np
    import pandas as pd
    N = len(feat_df)
    sim = similarity_matrix(feat_df, similarity, alpha)

    # kneighbors_graph 需要"距离"，用 -sim
    knn = kneighbors_graph(-sim, n_neighbors=min(k, N - 1), mode="connectivity",
                           include_self=False)
    adj = ((knn + knn.T) > 0).astype(float)  # 对称 0/1 邻接
    src, dst = adj.nonzero()
    w = sim[src, dst]
    # 按目标节点分组 softmax
    df_w = pd.DataFrame({"dst": dst, "w": w})
    df_w["w"] = df_w.groupby("dst")["w"].transform(lambda s: np.exp(s - s.max()))
    df_w["w"] = df_w["w"] / df_w.groupby("dst")["w"].transform("sum")
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(df_w["w"].values, dtype=torch.float)
    return edge_index, edge_weight


if __name__ == "__main__":
    from src.data.features import build_feature_matrix
    fm = build_feature_matrix(save=False)
    ei, ew = build_knn_graph(fm)
    print(f"节点 {fm.shape[0]}，边 {ei.shape[1]}")
