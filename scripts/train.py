# train.py (minimal-improvement version)
import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
from torch import nn
import ast

from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    AdamW,
    # roberta基线模型
    AutoModelForSequenceClassification,
)

# roberta+CNN模型
from model_cnn import RobertaTextCNNForMultiLabel

# roberta+注意力模型
from model_attention import RobertaAttentionForMultiLabel

# roberta+CNN与注意力拼接模型
from model_cnn_attention import RobertaCNNAttentionForMultiLabel

from config import Config
from dataset import PatentDataset
from metrics import compute_metrics

import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _safe_parse_labels(s: str):
    s = str(s).replace('""', '"')
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    try:
        x = json.loads(s)
        return x if isinstance(x, list) else []
    except Exception:
        try:
            x = ast.literal_eval(s)
            return x if isinstance(x, list) else []
        except Exception:
            return []

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping (过滤掉极易区分的负样本)
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # 概率衰减：注意 los_neg 里的 pow 计算使用的是 (1 - xs_neg)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps)) * torch.pow(1 - xs_pos, self.gamma_pos)
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps)) * torch.pow(1 - xs_neg, self.gamma_neg)

        loss = los_pos + los_neg
        return -loss.mean()
        
class HierarchyASLTrainer(Trainer):
    def __init__(self, penalty_matrix_path, sibling_discount=Config.SIBLING_DISCOUNT, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 加载之前的距离矩阵: 0=自身, 1=同主组兄弟, 2=跨主组远亲
        pm = torch.load(penalty_matrix_path)
        
        # 构建“兄弟掩码”：提取出距离为 1.0 的同主组兄弟关系矩阵 [num_labels, num_labels]
        self.sibling_mask = (pm == 1.0).float()
        
        # 兄弟误判的惩罚折扣 (例如 0.3 表示如果错认成兄弟，只承受 30% 的 loss 惩罚)
        self.sibling_discount = sibling_discount
        
        # ASL 的超参数
        self.gamma_neg = Config.GAMMA_NEG
        self.gamma_pos = Config.GAMMA_POS
        self.clip = Config.CLIP

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"].float()
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.get("logits")

        # --- 1. ASL 基础计算 ---
        x_sigmoid = torch.sigmoid(logits)
        xs_neg = 1 - x_sigmoid
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # 正样本 Loss (y=1) -> 保持不变
        los_pos = labels * torch.log(x_sigmoid.clamp(min=1e-8)) * torch.pow(1 - x_sigmoid, self.gamma_pos)

        # 负样本 Base Loss (y=0)
        los_neg_base = (1 - labels) * torch.log(xs_neg.clamp(min=1e-8)) * torch.pow(x_sigmoid, self.gamma_neg)

        # --- 2. 【核心创新】引入体系感知负样本折扣 ---
        sibling_mask = self.sibling_mask.to(logits.device)

        # 矩阵乘法：如果某个负标签是当前样本任一真实标签的“兄弟”，对应位置会大于 0
        # is_sibling_of_true 形状: [batch_size, num_labels]
        is_sibling_of_true = torch.matmul(labels, sibling_mask) > 0

        # 初始化负样本惩罚权重为全 1.0
        neg_weights = torch.ones_like(logits)
        # 将兄弟节点的惩罚权重打折 (降为 0.3)
        neg_weights[is_sibling_of_true] = self.sibling_discount

        # 应用体系折扣
        los_neg = los_neg_base * neg_weights

        # --- 3. 汇总求均值 ---
        loss = -(los_pos + los_neg).mean()

        return (loss, outputs) if return_outputs else loss
    
def calculate_pos_weight_from_csv(train_csv, label2id, device="cpu"):
    df = pd.read_csv(train_csv)
    pos = torch.zeros(len(label2id), dtype=torch.float32)

    for s in df["labels"].astype(str):
        codes = _safe_parse_labels(s)
        for c in set(codes):
            if c in label2id:
                pos[label2id[c]] += 1.0

    n = float(len(df))
    neg = n - pos
    pos = torch.clamp(pos, min=1.0)
    w = neg / pos
    return w.to(device)

class ASLTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # 移除 pos_weights 的接收，因为 ASL 不需要基于频率的全局权重
        super().__init__(*args, **kwargs)
        # 初始化 ASL，参数可调
        self.loss_fct = AsymmetricLoss(gamma_neg=Config.GAMMA_NEG, gamma_pos=Config.GAMMA_POS, clip=Config.CLIP)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"].float()
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.get("logits")

        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

class WeightedTrainer(Trainer):
    def __init__(self, pos_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weights = pos_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"].float()  # 不pop，避免副作用
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.get("logits")

        pw = self.pos_weights.to(logits.device)  # 设备对齐
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=pw)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


class MetricsLoggerCallback(EarlyStoppingCallback):
    """
    Extend EarlyStoppingCallback: also log metrics to jsonl after each evaluation.
    """
    def __init__(self, early_stopping_patience=3, output_dir="./output"):
        super().__init__(early_stopping_patience=early_stopping_patience)
        self.output_path = os.path.join(output_dir, "metrics_history.jsonl")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # call EarlyStoppingCallback logic
        super().on_evaluate(args, state, control, metrics=metrics, **kwargs)
        if metrics is None:
            return control
        rec = {"step": int(state.global_step), "epoch": float(state.epoch) if state.epoch is not None else None}
        rec.update({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})

        os.makedirs(args.output_dir, exist_ok=True)
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return control


def build_training_args():
    """
    Keep compatibility with different transformers versions:
    - some use evaluation_strategy, some accept eval_strategy
    We'll prefer evaluation_strategy, which is standard.
    """
    return TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        label_names=["labels"],  # <=== 新增这一行：强制告诉Trainer验证集标签的键名
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=Config.LR,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.EPOCHS,
        warmup_ratio=Config.WARMUP_RATIO,
        weight_decay=Config.WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="hr@3",  # make sure compute_metrics returns this key
        greater_is_better=True,
        fp16=True,
        remove_unused_columns=False,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
    )


def train():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    set_seed(getattr(Config, "SEED", 42))

    # 1) 读取固定标签空间
    with open(Config.LABEL_JSON, "r", encoding="utf-8") as f:
        labels = json.load(f)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    # 2) 保存本次运行配置（强烈建议）
    with open(os.path.join(Config.OUTPUT_DIR, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "TRAIN_CSV": Config.TRAIN_CSV,
            "VAL_CSV": Config.VAL_CSV,
            "TEST_CSV": getattr(Config, "TEST_CSV", ""),
            "LABEL_JSON": Config.LABEL_JSON,
            "MODEL_NAME": Config.MODEL_NAME,
            "NUM_LABELS": len(labels),
            "MAX_LEN": Config.MAX_LEN,
            "BATCH_SIZE": Config.BATCH_SIZE,
            "LR": Config.LR,
            "EPOCHS": Config.EPOCHS,
            "WARMUP_RATIO": Config.WARMUP_RATIO,
            "WEIGHT_DECAY": Config.WEIGHT_DECAY,
            "SEED": getattr(Config, "SEED", 42),
            "METRIC_FOR_BEST_MODEL": "hr@3",
            "GAMMA_NEG": Config.GAMMA_NEG,
            "GAMMA_POS": Config.GAMMA_POS,
            "CLIP": Config.CLIP,
            "SIBLING_DISCOUNT": Config.SIBLING_DISCOUNT,
            "MODEL_HEAD": getattr(Config, "MODEL_HEAD", "CNN"),
            "USE_SIBLING_PENALTY": getattr(Config, "USE_SIBLING_PENALTY", True),
        }, f, ensure_ascii=False, indent=2)

    # 3) 构建数据集
    train_ds = PatentDataset(Config.TRAIN_CSV, label2id, Config)
    val_ds = PatentDataset(Config.VAL_CSV, label2id, Config)

    # 4) 构建模型
    if getattr(Config, "MODEL_HEAD", "CNN") == "CNN":
        print("Model: RobertaTextCNNForMultiLabel (第四章完全体)")
        model = RobertaTextCNNForMultiLabel(Config)
        
    elif getattr(Config, "MODEL_HEAD", "ATTENTION") == "ATTENTION":
        # 【新增的逻辑分支】
        print("Model: Roberta + Attention Pooling (第三章消融实验用)")
        model = RobertaAttentionForMultiLabel(Config, num_labels=len(labels))
        
    elif getattr(Config, "MODEL_HEAD", "CNN") == "CNN+ATTENTION":
        print("Model: RobertaCNNAttentionForMultiLabel (CNN与ATTENTION双分支)")
        model = RobertaCNNAttentionForMultiLabel(Config)
        
    else:
        print("Model: Baseline RoBERTa (最原始的基线)")
        # 基线模型（纯纯的 RoBERTa 调包）
        model = AutoModelForSequenceClassification.from_pretrained(
            Config.MODEL_NAME,
            num_labels=len(labels),
            problem_type="multi_label_classification",
            label2id=label2id,
            id2label=id2label,
        )

    # 5) 计算 pos_weight（基于当前 train_csv）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("num_labels:", len(labels))
    print("Using Asymmetric Loss (ASL) - Skipping global pos_weight calculation.")
    
    
    # 使用Weighted BCE Loss
    # pos_weights = calculate_pos_weight_from_csv(
    #     Config.TRAIN_CSV,
    #     label2id,
    #     device=device
    # )

    # print(
    #     "pos_weight stats: min/median/max =",
    #     float(pos_weights.min().item()),
    #     float(pos_weights.median().item()),
    #     float(pos_weights.max().item())
    # )

    # 6) 训练参数与 Trainer
    args = build_training_args()

    # --- 新增：自定义分层学习率优化器 ---
    from torch.optim import AdamW
    
    # 假设你在 model_cnn.py 中，RoBERTa 骨干网络的变量名包含 "roberta" 
    # (例如 self.roberta = AutoModel.from_pretrained(...))
    # 我们为骨干网络保持 Config.LR (通常是 2e-5 或 3e-5)
    # 为非 roberta 的层 (也就是 TextCNN 和分类头) 分配 1e-3 的大学习率
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "roberta" in n],
            "lr": Config.LR, 
        },
        {
            "params": [p for n, p in model.named_parameters() if "roberta" not in n],
            "lr": 1e-3, 
        }
    ]
    
    # 实例化优化器，并应用 Config 中的权重衰减
    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=Config.WEIGHT_DECAY)
    # ------------------------------------

    # 1、使用ASL
    # trainer = ASLTrainer(
    #     model=model,
    #     args=args,
    #     train_dataset=train_ds,
    #     eval_dataset=val_ds,
    #     compute_metrics=compute_metrics,
    #     callbacks=[MetricsLoggerCallback(early_stopping_patience=3, output_dir=Config.OUTPUT_DIR)],
    #     optimizers=(optimizer, None),
    # )
    
    # 2、使用层级惩罚的ASL
    if getattr(Config, "USE_SIBLING_PENALTY", True):
        trainer = HierarchyASLTrainer(
            penalty_matrix_path="penalty_matrix.pt",
            sibling_discount=Config.SIBLING_DISCOUNT,
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[MetricsLoggerCallback(early_stopping_patience=2, output_dir=Config.OUTPUT_DIR)],
            optimizers=(optimizer, None),
        )
    else:
        trainer = ASLTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[MetricsLoggerCallback(early_stopping_patience=2, output_dir=Config.OUTPUT_DIR)],
            optimizers=(optimizer, None),
        )
    
    # 3、使用Weighted BCE Loss
    # trainer = WeightedTrainer(
    #     pos_weights=pos_weights,
    #     model=model,
    #     args=args,
    #     train_dataset=train_ds,
    #     eval_dataset=val_ds,
    #     compute_metrics=compute_metrics,
    #     callbacks=[MetricsLoggerCallback(early_stopping_patience=3, output_dir=Config.OUTPUT_DIR)],
    # )

    # 7) 开始训练并保存
    trainer.train()

    best_dir = os.path.join(Config.OUTPUT_DIR, "best_model_final")
    os.makedirs(best_dir, exist_ok=True)

    trainer.save_model(best_dir)  # 模型+config

    # 显式保存 tokenizer（强烈建议）
    train_ds.tokenizer.save_pretrained(best_dir)

    torch.save(model.state_dict(), os.path.join(best_dir, "pytorch_model.bin"))

    # 再落一份最终config快照（双保险）
    with open(os.path.join(best_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "TRAIN_CSV": Config.TRAIN_CSV,
            "VAL_CSV": Config.VAL_CSV,
            "TEST_CSV": getattr(Config, "TEST_CSV", ""),
            "LABEL_JSON": Config.LABEL_JSON,
            "MODEL_NAME": Config.MODEL_NAME,
            "NUM_LABELS": len(labels),
            "MAX_LEN": Config.MAX_LEN,
            "BATCH_SIZE": Config.BATCH_SIZE,
            "LR": Config.LR,
            "EPOCHS": Config.EPOCHS,
            "WARMUP_RATIO": Config.WARMUP_RATIO,
            "WEIGHT_DECAY": Config.WEIGHT_DECAY,
            "SEED": getattr(Config, "SEED", 42),
            "METRIC_FOR_BEST_MODEL": "hr@3",
            "MODEL_HEAD": getattr(Config, "MODEL_HEAD", "CNN"),
            "USE_SIBLING_PENALTY": getattr(Config, "USE_SIBLING_PENALTY", True)
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    train()
