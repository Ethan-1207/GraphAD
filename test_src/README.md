# test_src：同学实验沙盒

这是给你（同学）试**不同方法**的独立区域，后续会整体删掉，所以**不要改 `src/`**。

## 快速开始

```bash
# 在项目根目录
python -m test_src.my_experiments --quick      # 冒烟测试模板模型（约1分钟）
python -m test_src.my_experiments              # 完整训练
python -m src.experiments.run_baseline --model lr   # 先跑个基线感受一下数据规模
```

首次运行会自动构建特征矩阵并缓存到 `data/processed/`（约 1-2 分钟）。

## 你要做的事

1. 复制 `my_experiments.py`，改名成你的方法（如 `sage_v2.py`）；
2. 把里面的 `MyModel` 换成你的模型，只要求：
   - 有 `use_graph` 属性（True/False，决定是否用人群图）；
   - `forward(features, observed, edge_index, edge_weight, train_missing_rate)`
     返回 `(logits, {"recon_loss": ...})`；
3. 训练/评估用 `sandbox` 的 `sb.train` / `sb.evaluate`（统一划分、统一指标、统一缺失模拟）；
4. 结果自动存 `test_result/`。

## sandbox 速查

```python
import test_src.sandbox as sb

ctx = sb.load_data(seed=42, pet_missing_rate=0.0)  # 0/0.2/0.5/1.0 模拟PET缺失
model = sb.build_model(ctx, backbone="gat", k=20)  # 或换成你自己的模型
model, history = sb.train(model, ctx)
m = sb.evaluate(model, ctx, split="test")           # 或 "external"
print(sb.format_metrics(m))
```

## 注意

- **公平性**：数据划分、评估指标、PET 缺失场景模拟必须用沙盒提供的函数；
- **可复现**：每个实验固定 `--seed`，结果文件名带 seed；
- **别碰**：`result/`（主实验结果）、`src/`（主框架）。
