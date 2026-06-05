# Method Overview

- 任务定义：G06F 领域 500 类 CPC 小组多标签分类。
- 输入构造：`title + keywords + abstract`。
- DSP-ACL：Chinese RoBERTa、Attention Pooling、ASL。
- HA-MFF：在 DSP-ACL 上加入 TextCNN、Sibling Discount、HA-ASL。
- 评价指标：HitRate@k、Micro-F1@5。
- 模型目标：提升 Top-k 候选召回并改善同大组相似类别混淆。
