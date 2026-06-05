# Chinese Patent CPC Multi-Label Classification

本项目面向中文专利文档的 CPC 细粒度小组分类任务，针对 G06F 领域构建 500 类极端多标签分类场景。项目实现了 DSP-ACL 和 HA-MFF 两类模型，用于解决中文专利分类中的长文本语义表征、长尾标签失衡和 CPC 层级结构利用不足问题。

## 方法概览
- DSP-ACL：Chinese RoBERTa + Attention Pooling + Asymmetric Loss
- HA-MFF：Chinese RoBERTa + Attention Pooling + TextCNN + Sibling Discount + Hierarchy-Aware ASL

## 代码结构
- `configs/`：训练与实验配置。
- `src/`：模型与指标核心实现。
- `scripts/`：训练、评估、可视化与实验脚本。
- `scripts_data/`：数据清洗、筛选、切分与检查脚本。
- `data/`：样例数据与数据说明。
- `artifacts/`：实验所需静态工件（如层级折扣矩阵）。
- `results/`：论文结果汇总与可视化图片。
- `docs/`：方法、数据流水线与实验总结文档。

## 数据说明
仓库仅提供 sample 数据。复现实验请将完整数据放入 `data/`：
- `data/train_120.csv`
- `data/val_fixed.csv`
- `data/test_fixed.csv`
- `data/selected_labels.json`

数据字段示例：`title,keywords,abstract,labels`

## 环境安装
```bash
pip install -r requirements.txt
```

## 训练 DSP-ACL
```bash
python scripts/train.py --model dsp_acl --config configs/config.py
```

## 训练 HA-MFF
```bash
python scripts/generate_penalty_matrix.py
python scripts/train.py --model ha_mff --config configs/config.py
```

## 评估
```bash
python scripts/evaluate_test.py
```

## 注意力可视化
```bash
python scripts/plot_attention.py
```

## 主要实验结果
- DSP-ACL HitRate@10 = 84.99%
- HA-MFF HitRate@10 = 84.96%
- HA-MFF Micro-F1@5 = 33.59%

## Citation
```bibtex
@thesis{liao2026patent,
  title={面向中文专利文档的语义理解驱动多标签分类算法研究},
  author={廖乐源},
  year={2026},
  school={东北大学秦皇岛分校}
}
```
