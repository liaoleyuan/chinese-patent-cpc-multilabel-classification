# Data Notes

- 本仓库不直接包含完整中文专利数据。
- 公开仓库仅提供 `sample_train.csv`、`sample_val.csv`、`sample_test.csv`。
- 用户若要复现实验，需要将完整数据放置到 `data/` 目录下：
  - `data/train_120.csv`
  - `data/val_fixed.csv`
  - `data/test_fixed.csv`
- 标签文件 `selected_labels.json` 已提供，用于 500 类 CPC 小组映射。
