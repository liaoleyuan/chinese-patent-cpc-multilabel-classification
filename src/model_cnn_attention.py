"""HA-MFF model placeholder."""
import torch
import torch.nn as nn
from transformers import AutoModel
from model_attention import AttentionPooling


class RobertaCNNAttentionForMultiLabel(nn.Module):
    """
    RoBERTa + TextCNN + Attention Pooling Head
    """
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.NUM_LABELS
        self.hidden_size = getattr(config, "HIDDEN_SIZE", 768)

        self.roberta = AutoModel.from_pretrained(config.MODEL_NAME)

        # CNN Params
        self.kernel_sizes = getattr(config, "CNN_KERNEL_SIZES", [3, 4, 5])
        self.num_filters = getattr(config, "CNN_NUM_FILTERS", 256)
        self.dropout_p = getattr(config, "DROPOUT", 0.2)

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.num_filters,
                kernel_size=k
            )
            for k in self.kernel_sizes
        ])

        # Attention Pooling Params
        self.attn_pool = AttentionPooling(hidden_size=self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)

        # The concatenated feature dimension
        self.concat_dim = (self.num_filters * len(self.kernel_sizes)) + self.hidden_size
        self.classifier = nn.Linear(self.concat_dim, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.roberta(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        x = outputs.last_hidden_state  # [B, L, H]

        # --- Attention Pooling Path ---
        attn_feat, _ = self.attn_pool(x, attention_mask)

        # --- CNN Path ---
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x_cnn = x * mask  # PAD to 0
        else:
            x_cnn = x

        x_cnn = x_cnn.transpose(1, 2)  # [B, H, L]

        conv_feats = []
        for conv in self.convs:
            c = torch.relu(conv(x_cnn))
            p = torch.max(c, dim=2).values
            conv_feats.append(p)

        cnn_feat = torch.cat(conv_feats, dim=1)

        # --- Concat ---
        feat = torch.cat([cnn_feat, attn_feat], dim=1)
        feat = self.dropout(feat)

        logits = self.classifier(feat)

        return {"logits": logits}
