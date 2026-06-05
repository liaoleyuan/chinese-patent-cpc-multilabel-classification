# Data Pipeline

This document outlines the end-to-end preprocessing workflow used to construct the G06F 500-label extreme multi-label classification benchmark.

1. **Raw patent data cleaning:** Normalize fields, remove corrupted records, and standardize text encoding.
2. **G06F domain filtering:** Select patents whose CPC codes fall under the G06F section to form the target domain corpus.
3. **Document-level de-duplication / isolation:** Enforce document-level separation to avoid leakage and ensure reproducible splits.
4. **Greedy quota-based sampling:** Apply a greedy quota strategy to address long-tail label imbalance and obtain a controlled training subset.
5. **Split generation:** Produce `train_120.csv`, `val_fixed.csv`, and `test_fixed.csv`.
6. **Label space export:** Save the final 500-label space into `selected_labels.json` for consistent training/inference.
