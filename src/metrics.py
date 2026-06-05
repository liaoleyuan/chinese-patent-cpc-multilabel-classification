# metrics.py

"""
p@k / r@k / hr@k：完全不依赖阈值，更符合“推荐”场景
micro/macro_f1@k：用 top‑k 预测出来的二值矩阵算 F1，与 top‑k 指标口径一致
micro/macro_f1@0.5：传统多标签阈值法，用来补充说明（但在极不平衡时可能偏低/不稳定）
"""
import numpy as np

def _safe_div(a, b, eps=1e-9):
    return a / (b + eps)

def _f1_from_counts(tp, fp, fn, eps=1e-9):
    precision = _safe_div(tp, tp + fp, eps)
    recall = _safe_div(tp, tp + fn, eps)
    return _safe_div(2 * precision * recall, precision + recall, eps)

def compute_metrics(eval_pred):
    """
    Returns a dict of metrics for multi-label classification.
    - Top-K metrics: P@K, R@K, HR@K, micro/macro F1@K
    - Threshold metrics: micro/macro F1@0.5
    """
    logits, labels = eval_pred
    # Some models return extra outputs (e.g., attn_weights). Keep logits only.
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    # logits: (N, L), labels: (N, L)
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    y_true = labels.astype(np.int32)

    n, num_labels = probs.shape

    results = {}

    # ---- Basic stats (useful for sanity & paper) ----
    true_pos_per_sample = y_true.sum(axis=1)
    results["avg_true_labels"] = float(true_pos_per_sample.mean())
    results["pct_zero_true_labels"] = float((true_pos_per_sample == 0).mean())

    # ---- Top-K metrics ----
    # For 10-label experiment, @1 is important; for 500-label, @10 can be useful.
    K_LIST = [1, 3, 5, 10]

    for k in K_LIST:
        kk = min(k, num_labels)
        # topk indices per sample
        topk_idx = np.argpartition(-probs, kk-1, axis=1)[:, :kk]  # unsorted top-kk
        y_pred = np.zeros_like(y_true, dtype=np.int32)
        rows = np.arange(n)[:, None]
        y_pred[rows, topk_idx] = 1

        tp = (y_pred & y_true).sum(axis=1)           # per-sample tp
        pred_pos = y_pred.sum(axis=1)                # = kk
        actual_pos = true_pos_per_sample

        results[f"p@{k}"] = float((tp / kk).mean())
        results[f"r@{k}"] = float(_safe_div(tp, actual_pos).mean())
        results[f"hr@{k}"] = float((tp > 0).mean())
        results[f"avg_pred_labels@{k}"] = float(pred_pos.mean())

        # micro F1@K
        TP_micro = int(((y_pred == 1) & (y_true == 1)).sum())
        FP_micro = int(((y_pred == 1) & (y_true == 0)).sum())
        FN_micro = int(((y_pred == 0) & (y_true == 1)).sum())
        results[f"micro_f1@{k}"] = float(_f1_from_counts(TP_micro, FP_micro, FN_micro))

        # macro F1@K (average F1 over labels)
        TP_l = ((y_pred == 1) & (y_true == 1)).sum(axis=0)
        FP_l = ((y_pred == 1) & (y_true == 0)).sum(axis=0)
        FN_l = ((y_pred == 0) & (y_true == 1)).sum(axis=0)
        f1_l = _f1_from_counts(TP_l, FP_l, FN_l)
        results[f"macro_f1@{k}"] = float(np.mean(f1_l))

    # ---- Threshold=0.5 metrics (classic multilabel) ----
    y_pred_thr = (probs >= 0.5).astype(np.int32)

    TP_micro = int(((y_pred_thr == 1) & (y_true == 1)).sum())
    FP_micro = int(((y_pred_thr == 1) & (y_true == 0)).sum())
    FN_micro = int(((y_pred_thr == 0) & (y_true == 1)).sum())
    results["micro_f1@0.5"] = float(_f1_from_counts(TP_micro, FP_micro, FN_micro))

    TP_l = ((y_pred_thr == 1) & (y_true == 1)).sum(axis=0)
    FP_l = ((y_pred_thr == 1) & (y_true == 0)).sum(axis=0)
    FN_l = ((y_pred_thr == 0) & (y_true == 1)).sum(axis=0)
    f1_l = _f1_from_counts(TP_l, FP_l, FN_l)
    results["macro_f1@0.5"] = float(np.mean(f1_l))

    # also report how many labels predicted under threshold (sparseness)
    results["avg_pred_labels@0.5"] = float(y_pred_thr.sum(axis=1).mean())

    return results
