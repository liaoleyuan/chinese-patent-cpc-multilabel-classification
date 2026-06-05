# Experiment Results

基于 `results/all_experiments_results.csv` 与 `results/attention_barplot.png`：

- DSP-ACL 消融实验显示注意力池化与 ASL 组合在 Top-k 召回上表现稳定。
- HA-MFF 相比基础结构在同大组标签区分上更稳健。
- 注意力池化可视化展示了模型对关键技术术语与上下文片段的聚焦。
- Sibling Discount 有助于缓解同层级近邻标签间的混淆，提升层级一致性。
