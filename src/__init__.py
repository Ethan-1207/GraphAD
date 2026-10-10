"""GraphAD: 人群图 GNN 预测 MCI→AD 转化（缺失模态鲁棒）。

核心卖点：病人缺 PET 时，从"相似的病人"（人群图消息传递）借判断，
而不是做 PET 补全。数据：ADNI（ADNIMERGE2 R 数据包，.rda 格式）。
"""

__version__ = "0.1.0"
