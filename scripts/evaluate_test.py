# evaluate_test.py
import os
import json
import torch
from transformers import TrainingArguments
# 移除原生 Trainer，导入自定义的 Trainer
from train import HierarchyASLTrainer 

from config import Config
from dataset import PatentDataset
from metrics import compute_metrics
from model_cnn import RobertaTextCNNForMultiLabel
from model_attention import RobertaAttentionForMultiLabel
from model_cnn_attention import RobertaCNNAttentionForMultiLabel
from train import ASLTrainer


def evaluate_on_test():
    print("=== 开始最终测试集评估===")

    # 1) 加载标签
    with open(Config.LABEL_JSON, "r", encoding="utf-8") as f:
        labels = json.load(f)
    label2id = {l: i for i, l in enumerate(labels)}

    # 2) 测试集
    test_csv = getattr(Config, "TEST_CSV", None)
    if not test_csv:
        raise ValueError("Config.TEST_CSV 未设置，请在 config.py 中设置测试集路径。")
    test_ds = PatentDataset(test_csv, label2id, Config)
    print("Test CSV:", test_csv)
    print("num_labels:", len(labels))

    # 3) 加载模型（兼容 bin 和 safetensors）
    model_dir = os.path.join(Config.OUTPUT_DIR, "best_model_final")
    
    if getattr(Config, "MODEL_HEAD", "CNN") == "CNN":
        print("评估模式: RobertaTextCNNForMultiLabel (第四章完全体)")
        model = RobertaTextCNNForMultiLabel(Config)
        
    elif getattr(Config, "MODEL_HEAD", "ATTENTION") == "ATTENTION":
        print("评估模式: Roberta + Attention Pooling (第三章消融实验)")
        model = RobertaAttentionForMultiLabel(Config, num_labels=len(labels))
        
    elif getattr(Config, "MODEL_HEAD", "CNN") == "CNN+ATTENTION":
        print("评估模式: RobertaCNNAttentionForMultiLabel")
        model = RobertaCNNAttentionForMultiLabel(Config)
        
    else:
        print("评估模式: Baseline RoBERTa (基线模型)")
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            Config.MODEL_NAME,
            num_labels=len(labels),
            problem_type="multi_label_classification",
            label2id=label2id,
            id2label={i: l for l, i in label2id.items()}
        )
    
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    sf_path = os.path.join(model_dir, "model.safetensors")

    if os.path.exists(bin_path):
        print(f"找到 .bin 权重文件，正在加载: {bin_path}")
        state = torch.load(bin_path, map_location="cpu")
    elif os.path.exists(sf_path):
        print(f"找到 .safetensors 权重文件，正在加载: {sf_path}")
        from safetensors.torch import load_file
        state = load_file(sf_path)
    else:
        raise FileNotFoundError(f"未在 {model_dir} 找到 pytorch_model.bin 或 model.safetensors")

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("missing keys:", len(missing))
    print("unexpected keys:", len(unexpected))

    # 设备判断
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 4) 评估参数
    use_fp16 = torch.cuda.is_available()
    eval_args = TrainingArguments(
        output_dir=os.path.join(Config.OUTPUT_DIR, "test_results"),
        per_device_eval_batch_size=Config.BATCH_SIZE,
        fp16=use_fp16,
        report_to="none",
        remove_unused_columns=False,  
        # 显式告诉 Trainer 标签字段叫 "labels"，强制它收集标签并触发指标计算
        label_names=["labels"],
    )

    # 5) 根据测试的模型选择正确的 Trainer
    if getattr(Config, "MODEL_HEAD", "CNN") == "ATTENTION" or getattr(Config, "MODEL_HEAD", "CNN") == "BASELINE":
        # 如果是第三章的基线或 Attention 实验，使用纯 ASLTrainer
        print("Trainer: ASLTrainer (无层级惩罚)")
        trainer = ASLTrainer(
            model=model,
            args=eval_args,
            compute_metrics=compute_metrics,
        )
    else:
        # 如果是第四章的 CNN 实验，或者 CNN+ATTENTION 实验，使用带层级折扣的 HierarchyASLTrainer
        print("Trainer: HierarchyASLTrainer (开启 Sibling Discount)")
        trainer = HierarchyASLTrainer(
            penalty_matrix_path="penalty_matrix.pt",
            sibling_discount=Config.SIBLING_DISCOUNT,
            model=model,
            args=eval_args,
            compute_metrics=compute_metrics,
        )

    print("正在测试集上进行推理预测...")
    results = trainer.evaluate(eval_dataset=test_ds)

    # 6) 打印结果
    print("\n" + "=" * 60)
    print("          测试集最终评估报告 (Final Test Result)")
    print("=" * 60)
    print(f"{'Metric':<26} | {'Value':<12}")
    print("-" * 60)
    for key in sorted(results.keys()):
        if key.startswith("eval_"):
            metric_name = key.replace("eval_", "")
            val = results[key]
            if isinstance(val, (int, float)):
                print(f"{metric_name:<26} | {val:.6f}")
            else:
                print(f"{metric_name:<26} | {val}")
    print("=" * 60)

    # 7) 保存结果
    out_json = os.path.join(Config.OUTPUT_DIR, "final_test_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("结果已保存至:", out_json)


if __name__ == "__main__":
    evaluate_on_test()
