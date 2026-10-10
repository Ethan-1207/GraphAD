# -*- coding: utf-8 -*-
"""GNN 骨干：GCN / GAT / GraphSAGE（依赖 torch_geometric，惰性导入）。"""

import torch
import torch.nn as nn


def _build_conv(backbone: str, in_dim: int, out_dim: int):
    if backbone == "gcn":
        from torch_geometric.nn import GCNConv
        return GCNConv(in_dim, out_dim)
    if backbone == "gat":
        from torch_geometric.nn import GATConv
        return GATConv(in_dim, out_dim, heads=4, concat=False)
    if backbone == "sage":
        from torch_geometric.nn import SAGEConv
        return SAGEConv(in_dim, out_dim)
    raise ValueError(f"未知骨干: {backbone}")


class GNNEncoder(nn.Module):
    """k 层消息传递（1-3 层可调），输出节点嵌入。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 backbone: str = "gcn", num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        if not 1 <= num_layers <= 3:
            raise ValueError("消息传递层数限 1-3（论文消融设定）")
        self.dropout = dropout
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(
            [_build_conv(backbone, dims[i], dims[i + 1]) for i in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims[1:]])

    def forward(self, x, edge_index, edge_weight=None):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index, edge_weight)
            x = norm(x)
            x = x.relu()
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x
