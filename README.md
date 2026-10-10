# GraphAD: 人群图神经网络预测 MCI→AD 转化（缺失 PET 鲁棒）

> 核心问题：**病人缺 PET 时，能不能从"和他相似的病人"那里借到判断？**
> 现有方法要么缺 PET 直接用不了，要么靠"补全"硬凑（ITCFN / MCNet / CRAD）。
> 我们不做补全——让相似病人在人群图上**消息传递互相投票**，验证能否补上缺口。

预测任务：基线 MCI 患者 **3 年内是否转化为 AD**（二分类），ADNI 公开数据。

## 方法（四模块）

1. **模态编码层**：MRI（FreeSurfer 皮质厚度 + 海马体积）、PET（FDG/AV45 SUVR）、
   认知量表（MMSE / ADAS-Cog-13 / CDR-SB）分别 FC 映射到统一嵌入空间；
2. **模态掩码 + 跨模态补全**（辅助）：训练时随机遮 PET，用其余模态重建（辅助损失）；
3. **人群图消息传递**（核心卖点）：病人为节点，按表型相似度（认知量表）+ 基因相似度
   （APOE ε4）建 KNN 图，GCN / GAT / GraphSAGE 传播；
4. **分类头**：readout + MLP 输出转化概率。损失 = 交叉熵 + λ·重建损失。

## 数据

ADNI（[ADNIMERGE2](https://adni.loni.usc.edu/) R 数据包，`.rda` 格式，需自行申请下载）。

| 论文需求 | 数据来源 |
|---|---|
| 诊断标签 | `DXSUM`（基线 MCI，36 个月内转 Dementia = 1） |
| 认知量表 | `MMSE` / `ADAS`(TOTAL13) / `CDR`(CDRSB) |
| MRI 特征 | `ADSP_PHC_T1_FS`（73 维 combat 协调化厚度 + 海马 + ICV） |
| PET（允许缺失） | `UCBERKELEYFDG_8mm`(FDG MetaROI SUVR)、`UCBERKELEY_AMY_6MM`(AV45 SUVR) |
| 协变量 | `DM`(年龄)、`PTDEMOG`(性别/教育)、`APOERES`(APOE 基因型) |

实测基线缺失率（论文"模态缺失先验"）：FDG-PET 33%、AV45 36%、MRI 9.7%、量表 <2%。
注：AV45 仅 ADNI2+ 采集，ADNI1 训练集内全缺失，属预期行为。
（PRS 暂不可用，基因相似度用 APOE ε4 携带状态 + 等位基因数替代。）

**ADNI 数据受 Data Use Agreement 约束，本仓库不含任何数据文件。**

## 快速开始

```bash
pip install -r requirements.txt

# 数据管线（首次自动构建特征矩阵并缓存到 data/processed/）
python -m src.data.features

# 传统基线
python -m src.experiments.run_baseline --model all

# GraphAD 主模型（test + 外部验证一次跑完）
python -m src.experiments.run_graphad

# 全部消融（去图/去补全/GCN vs GAT vs SAGE/k 值/相似度/PET 缺失 0-100%）
python -m src.experiments.run_ablation          # --quick 冒烟
```

## 实验协议（防泄漏）

- ADNI-1 按受试者分层 7:1:2 → train / val / test（同一人所有访视只进同一折）；
- ADNI-GO / 2 / 3 / 4 整体作为**独立外部队列**；
- 人群图采用 Parisot(2017) 转导式设定：连边只用特征不用标签；
- 固定随机种子，配置随结果落盘 `result/*.json`；
- 指标：ACC / AUC / 敏感性 / 特异性 / F1 + bootstrap 95% CI。

## 代码结构

```
src/
├── config.py               # 全部实验配置（协议、特征定义、超参）
├── data/
│   ├── loader.py           # 读取 .rda（pyreadr）
│   ├── cohort.py           # 基线 MCI 队列 + 3 年转化标签
│   ├── features.py         # 三模态 + 协变量基线特征（按日期对齐）
│   └── split.py            # 受试者级 7:1:2 + 外部队列
├── graph/build_graph.py    # KNN 人群图（表型/基因/混合相似度）
├── models/
│   ├── encoders.py         # 模态编码 + 掩码
│   ├── imputer.py          # 跨模态注意力补全（辅助）
│   ├── gnn.py              # GCN / GAT / GraphSAGE
│   ├── graphad.py          # 全模型组装
│   └── baselines.py        # LR / RF / SVM
├── training/               # 训练循环、评估（含 95% CI）、种子
└── experiments/            # 入口脚本：baseline / graphad / ablation
test_src/                   # 实验沙盒（试新方法用，见其 README）
```

## 引用

如使用本代码，请引用 ADNI 数据库并引用人群图框架：
Parisot et al., *Disease Prediction using Graph Convolutional Networks*, MICCAI 2017.
