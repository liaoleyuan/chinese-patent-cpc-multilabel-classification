# model_attention.py
import torch
import torch.nn as nn
from transformers import AutoModel

class AttentionPooling(nn.Module):
    """
    第三章新增的架构：注意力池化模块
    用于解决长文本下 [CLS] 向量的特征遗忘问题
    """
    def __init__(self, hidden_size=768):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: [batch_size, seq_len]
        
        # 1. 计算每个 token 的注意力打分
        attn_weights = self.attention(hidden_states) 
        attn_weights = attn_weights.squeeze(-1) # [batch_size, seq_len]
        
        # 2. Mask 处理：把 padding 的部分变成极小值，确保 softmax 后权重为 0
        extended_mask = (1.0 - attention_mask) * -10000.0
        attn_weights = attn_weights + extended_mask
        
        # 3. 归一化得到概率分布
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # 4. 加权求和，得到融合了全局信息的稠密向量
        pooled_output = torch.sum(hidden_states * attn_weights.unsqueeze(-1), dim=1)
        
        return pooled_output, attn_weights # <--- 修改这里，返回 attn_weights

class RobertaAttentionForMultiLabel(nn.Module):
    """
    第三章的完全体模型：RoBERTa + Attention Pooling
    """
    def __init__(self, config, num_labels):
        super(RobertaAttentionForMultiLabel, self).__init__()
        # 加载 RoBERTa 骨干网络
        self.roberta = AutoModel.from_pretrained(config.MODEL_NAME)
        
        # 加载我们自己写的 Attention Pooling 层
        self.attn_pool = AttentionPooling(hidden_size=self.roberta.config.hidden_size)
        self.dropout = nn.Dropout(self.roberta.config.hidden_dropout_prob)
        
        # 最终的线性分类头映射到 500 类
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        # 1. 过 RoBERTa，获取所有 token 的输出
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs.last_hidden_state # [batch_size, seq_len, 768]
        
        # 2. 使用 Attention Pooling 替代原来粗糙的 [CLS]
        pooled_output, attn_weights = self.attn_pool(sequence_output, attention_mask)
        pooled_output = self.dropout(pooled_output)
        
        # 3. 输出多标签分类的 logits
        logits = self.classifier(pooled_output)
        
        # 注意：为了兼容你自定义的 Trainer，这里返回包含 logits 的字典
        return {"logits": logits, "attn_weights": attn_weights}
