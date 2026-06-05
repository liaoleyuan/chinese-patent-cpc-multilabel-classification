import os
import json
import numpy as np
import torch
from safetensors.torch import load_file
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import PatentDataset

def main(target_dir):
    Config.OUTPUT_DIR = target_dir
    model_dir = os.path.join(target_dir, "best_model_final")
    sf_path = os.path.join(model_dir, "model.safetensors")
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    
    if os.path.exists(sf_path):
        state = load_file(sf_path)
    else:
        state = torch.load(bin_path, map_location="cpu")

    with open(Config.LABEL_JSON, "r", encoding="utf-8") as f:
        labels = json.load(f)
    label2id = {l: i for i, l in enumerate(labels)}

    dataset = PatentDataset(Config.TEST_CSV, label2id, Config)
    
    # Custom collate_fn handling dataset return
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch]) if "labels" in batch[0] else None
        }

    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

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
            Config.MODEL_NAME, num_labels=500)

    model.load_state_dict(state, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if batch["labels"] is not None:
                all_labels.append(batch["labels"].cpu().numpy())
            
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            # handle dict output
            if isinstance(output, dict) and "logits" in output:
                logits = output["logits"]
            elif hasattr(output, "logits"):
                logits = output.logits
            else:
                logits = output[0]
                
            all_logits.append(logits.cpu().numpy())

    np.save(os.path.join(target_dir, "val_logits.npy"), np.vstack(all_logits))
    if len(all_labels) > 0:
        np.save(os.path.join(target_dir, "val_labels.npy"), np.vstack(all_labels))
    print(f"Done for {target_dir}")

if __name__ == "__main__":
    main(sys.argv[1])
