import torch
import torch.nn as nn
from transformers import AutoModel


class RobertaTextCNNForMultiLabel(nn.Module):
    """
    RoBERTa + TextCNN Head for multi-label classification
    - Encoder: AutoModel (e.g., Chinese RoBERTa)
    - Head: Conv1d(k=3,4,5) + GlobalMaxPool + FC
    - Loss: 由外部 Trainer (ASL) 接管
    """
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.NUM_LABELS
        self.hidden_size = getattr(config, "HIDDEN_SIZE", 768)

        # 【核心修改 1】: 命名改为 roberta，确保 train.py 里的分层学习率能正确识别！
        self.roberta = AutoModel.from_pretrained(config.MODEL_NAME)

        # cnn head params
        self.kernel_sizes = getattr(config, "CNN_KERNEL_SIZES", [3, 4, 5])
        self.num_filters = getattr(config, "CNN_NUM_FILTERS", 256)
        self.dropout_p = getattr(config, "DROPOUT", 0.2)

        # Conv1d expects: [B, C, L]
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.num_filters,
                kernel_size=k
            )
            for k in self.kernel_sizes
        ])

        self.dropout = nn.Dropout(self.dropout_p)
        self.classifier = nn.Linear(self.num_filters * len(self.kernel_sizes), self.num_labels)
        
        # 【修改 3】: 删除了 self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # 提取骨干特征
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # [B, L, H]

        # 【核心修改 2】: 引入 Attention Mask，将 PAD token 的特征清零
        # attention_mask 形状为 [B, L]，我们需要将其扩展为 [B, L, 1] 以便与 x 广播相乘
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = x * mask  # PAD 位置的特征全部变成 0

        # 转置为 [B, H, L] 以适应 Conv1d 的输入格式
        x = x.transpose(1, 2)

        conv_feats = []
        for conv in self.convs:
            # 卷积 + 激活: [B, F, L-k+1]
            c = torch.relu(conv(x))
            # 沿着时间维度进行全局最大池化 -> [B, F]
            p = torch.max(c, dim=2).values
            conv_feats.append(p)

        # 拼接所有卷积核提取的特征: [B, F * len(K)]
        feat = torch.cat(conv_feats, dim=1)
        feat = self.dropout(feat)

        logits = self.classifier(feat)  # [B, num_labels]

        # 直接返回 logits，将 Loss 的计算完全交给外部的 HierarchyASLTrainer
        return {"logits": logits}
