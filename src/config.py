# -*- coding: utf-8 -*-
"""全局配置：路径、划分协议、特征定义、图构建参数。"""

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "rawdata_ADNIMERGE2" / "data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULT_DIR = PROJECT_ROOT / "result"
TEST_RESULT_DIR = PROJECT_ROOT / "test_result"

# ---------------------------------------------------------------------------
# 实验协议（对应飞书实验设计）
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# 标签：基线 MCI，LABEL_HORIZON_MONTHS 个月内转为 Dementia 记为 1
LABEL_HORIZON_MONTHS = 36

# 数据划分：ADNI-1 按受试者分层 7:1:2；ADNI-GO/2/3(/4) 作为独立外部队列
DEV_PHASES = ("ADNI1",)
EXTERNAL_PHASES = ("ADNIGO", "ADNI2", "ADNI3", "ADNI4")
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.1, 0.2

# 基线特征匹配窗口：以基线诊断日期为锚，±N 月内取最近的检查
BASELINE_MATCH_WINDOW_MONTHS = 6

# ---------------------------------------------------------------------------
# 特征定义（基于实际 .rda 字段核查，2026-10-10）
# ---------------------------------------------------------------------------
# 认知量表：MMSE.rda/MMSCORE, ADAS.rda/TOTAL13(ADAS-Cog-13), CDR.rda/CDRSB
COG_TABLES = {
    "MMSE": "MMSCORE",
    "ADAS": "TOTAL13",
    "CDR": "CDRSB",
}
COG_FEATURES = ["mmse", "adas13", "cdrsb"]

# MRI：ADSP_PHC_T1_FS.rda，FreeSurfer 皮质厚度（combat 协调化）+ 海马体积 + ICV
MRI_TABLE = "ADSP_PHC_T1_FS"
MRI_THICKNESS_SUFFIX = "_thickness_combat"  # 运行时按此后缀动态发现 140 个厚度列
MRI_HIPPO_COLS = ["Left_Hippocampus_combat", "Right_Hippocampus_combat"]
MRI_ICV_COL = "EstimatedTotalIntraCranialVol_combat"

# PET（允许缺失，缺失本身是论文卖点的一部分）
PET_FDG_TABLE = "UCBERKELEYFDG_8mm"       # 长表: ROINAME='MetaROI' → MEAN 为 SUVR
PET_FDG_ROI = "MetaROI"
PET_AV45_TABLE = "UCBERKELEY_AMY_6MM"     # SUMMARY_SUVR（淀粉样 PET）
PET_FEATURES = ["pet_fdg_suvr", "pet_av45_suvr"]

# 协变量：年龄/性别/教育（DM+PTDEMOG），APOE ε4（APOERES 基因型解析）
COVARIATES = ["cov_age", "cov_sex_male", "cov_educ", "cov_apoe4_carrier", "cov_apoe4_count"]

# ---------------------------------------------------------------------------
# 人群图构建（核心卖点）
# ---------------------------------------------------------------------------
GRAPH_K = 10                      # KNN 的 k
GRAPH_SIMILARITY = "mixed"        # phenotype | gene | mixed
GRAPH_PHENO_COLS = COG_FEATURES   # 表型相似度：认知量表 z 距离
GRAPH_GENE_COLS = ["cov_apoe4_carrier", "cov_apoe4_count"]  # 基因相似度（无 PRS，用 APOE）
GRAPH_MIXED_ALPHA = 0.7           # mixed: alpha*表型 + (1-alpha)*基因

# ---------------------------------------------------------------------------
# 模型默认超参（CLI 可覆盖）
# ---------------------------------------------------------------------------
EMB_DIM = 64
GNN_HIDDEN = 64
GNN_LAYERS = 2
GNN_BACKBONE = "gcn"              # gcn | gat | sage
DROPOUT = 0.5
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 300
PATIENCE = 30
RECON_LAMBDA = 0.1                # 辅助重建损失权重
TRAIN_MISSING_RATE = 0.3          # 训练时随机遮 PET 的比例（模态掩码）


@dataclass
class ExperimentConfig:
    """一次实验的完整配置，训练时随结果一起落盘，保证可复现。"""
    name: str = "graphad"
    seed: int = RANDOM_SEED
    label_horizon_months: int = LABEL_HORIZON_MONTHS
    k: int = GRAPH_K
    similarity: str = GRAPH_SIMILARITY
    backbone: str = GNN_BACKBONE
    gnn_layers: int = GNN_LAYERS
    emb_dim: int = EMB_DIM
    gnn_hidden: int = GNN_HIDDEN
    dropout: float = DROPOUT
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    max_epochs: int = MAX_EPOCHS
    patience: int = PATIENCE
    recon_lambda: float = RECON_LAMBDA
    train_missing_rate: float = TRAIN_MISSING_RATE
    # 消融开关
    use_graph: bool = True       # False = 去人群图（退化成普通 MLP）
    use_imputer: bool = True     # False = 去模态补全
    eval_pet_missing_rate: float = 0.0  # 评估时 PET 缺失率 0/0.2/0.5/1.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)
