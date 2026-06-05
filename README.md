# Chinese Patent CPC Multi-Label Classification

This project studies CPC fine-grained subgroup classification for Chinese patent documents. We focus on the G06F domain and construct an extreme multi-label classification task with 500 CPC subgroup labels. The repository implements two models—DSP-ACL and HA-MFF—targeting long-document semantic representation, long-tail label imbalance, and better utilization of the CPC hierarchy.

## Method Overview
- DSP-ACL: Chinese RoBERTa + Attention Pooling + Asymmetric Loss (ASL)
- HA-MFF: Chinese RoBERTa + Attention Pooling + TextCNN + Sibling Discount + Hierarchy-Aware ASL

## Repository Structure
- `configs/`: Training and experiment configuration.
- `src/`: Core model and metrics implementations.
- `scripts/`: Training, evaluation, visualization, and experiment scripts.
- `scripts_data/`: Data cleaning, filtering, splitting, and sanity-check scripts.
- `data/`: Sample data and dataset notes.
- `artifacts/`: Static artifacts required by experiments (e.g., hierarchy penalty/discount matrix).
- `results/`: Paper result summaries and figures.
- `docs/`: Method, data pipeline, and experiment summary docs.

## Data
This repository only includes **sample** data. To reproduce experiments locally, place the full datasets under `data/`:
- `data/train_120.csv`
- `data/val_fixed.csv`
- `data/test_fixed.csv`
- `data/selected_labels.json`

### Dataset Source
The dataset used in this project comes from:

Liu Q, Bao H F, Zhang J M, et al. *CN-US-EU-JP-Patent_2020-2025: version 1.0* [DS/OL]. GitHub, 2025. https://github.com/liuquan-ustc-qmai/CN-US-EU-JP-Patent_2020-2025.

Example fields: `title,keywords,abstract,labels`

## Installation
```bash
pip install -r requirements.txt
```

## Train DSP-ACL
```bash
# change the "MODEL_HEAD" value to "ATTENTION" in config.py before running the command below
python scripts/train.py --config configs/config.py
```

## Train HA-MFF
```bash
python scripts/generate_penalty_matrix.py
# change the "MODEL_HEAD" value to "CNN+ATTENTION" before running the command below
python scripts/train.py --config configs/config.py
```

## Evaluate
```bash
python scripts/evaluate_test.py
```

## Attention Visualization
```bash
python scripts/plot_attention.py
```

## Main Results
- DSP-ACL HitRate@10 = 84.99%
- HA-MFF HitRate@10 = 84.96%
- HA-MFF Micro-F1@5 = 33.59%

## Citation
```bibtex
@thesis{liao2026patent,
  title={Research on Semantic Understanding-Driven Multi-Label Classification Algorithm for Chinese Patent Documents},
  author={Leyuan Liao},
  year={2026},
  school={Northeastern University at Qinhuangdao}
}
```
