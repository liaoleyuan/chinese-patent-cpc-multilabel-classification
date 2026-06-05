import os
import json
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import Trainer, TrainingArguments
import sys

from config import Config
from dataset import PatentDataset
from model_cnn_attention import RobertaCNNAttentionForMultiLabel

def dump_for_dir(target_dir):
    Config.OUTPUT_DIR = target_dir
    model_dir = os.path.join(target_dir, "best_model_final")
    sf_path = os.path.join(model_dir, "model.safetensors")
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    
    if os.path.exists(sf_path):
        state = load_file(sf_path)
    elif os.path.exists(bin_path):
        state = torch.load(bin_path, map_location="cpu")
    else:
        print(f"Weight not found in {model_dir}")
        return

    with open(Config.LABEL_JSON, "r", encoding="utf-8") as f:
        labels = json.load(f)
    label2id = {l: i for i, l in enumerate(labels)}

    # 用测试集做评估更贴合"Bad Case发现"
    # 这里我们使用 TEST_CSV 因为通常是测试集用来找 bad case 和做统计
    dataset = PatentDataset(Config.TEST_CSV, label2id, Config)
    
    # 根据模型目录名判断使用哪种 Head
    if "CNN" in target_dir and "ATTENTION" in target_dir:
        from model_cnn_attention import RobertaCNNAttentionForMultiLabel
        model = RobertaCNNAttentionForMultiLabel(Config)
    elif "CNN" in target_dir:
        from model_cnn import RobertaTextCNNForMultiLabel
        model = RobertaTextCNNForMultiLabel(Config)
    elif "ATTENTION" in target_dir:
        from model_attention import RobertaAttentionForMultiLabel
        model = RobertaAttentionForMultiLabel(Config, num_labels=len(labels))
    else:
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            Config.MODEL_NAME, num_labels=500, label2id=label2id, id2label={i:l for l,i in label2id.items()}, problem_type="multi_label_classification")

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[{target_dir}] missing:{len(missing)} unexpected:{len(unexpected)}")

    model = model.to(Config.DEVICE)

    args = TrainingArguments(
        output_dir=os.path.join(target_dir, "val_dump"),
        per_device_eval_batch_size=Config.BATCH_SIZE,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"]
    )

    class PredTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels", None)
            outputs = model(input_ids=inputs.get("input_ids"), attention_mask=inputs.get("attention_mask"))
            logits = outputs.get("logits") if isinstance(outputs, dict) else outputs[0]
            loss = torch.tensor(0.0, device=logits.device)
            return (loss, outputs) if return_outputs else loss

    trainer = PredTrainer(model=model, args=args)
    pred = trainer.predict(dataset)

    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    labels_np = pred.label_ids

    np.save(os.path.join(target_dir, "val_logits.npy"), logits)
    np.save(os.path.join(target_dir, "val_labels.npy"), labels_np)
    print(f"[{target_dir}] saved val_logits.npy {logits.shape}")
    print(f"[{target_dir}] saved val_labels.npy {labels_np.shape}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dump_for_dir(sys.argv[1])
    else:
        dump_for_dir("output_roberta_ASL_ATTENTION_penalty_CNN")
