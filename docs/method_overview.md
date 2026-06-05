# Method Overview

- **Task definition:** Extreme multi-label classification of **500 CPC sub-group labels** in the **G06F** domain for Chinese patent documents.
- **Input formulation:** Concatenate `title + keywords + abstract` as the textual input.
- **DSP-ACL:** Chinese RoBERTa with **attention pooling** and **Asymmetric Loss (ASL)**.
- **HA-MFF:** Extends DSP-ACL by adding a **TextCNN** local-term channel, a **Sibling Discount** hierarchy-aware penalty matrix, and **hierarchy-aware ASL (HA-ASL)**.
- **Evaluation metrics:** HitRate@k and Micro-F1@5.
- **Model objective:** Improve **Top-k candidate recall** and reduce confusions among semantically similar labels within the same higher-level CPC group, while encouraging hierarchical consistency.
