import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer

# 导入你配置和模型
from config import Config
from model_attention import RobertaAttentionForMultiLabel

def plot_attention_heatmap():
    # 1. 初始化模型与分词器
    # 这里需要填入你真实的标签数量
    num_labels = 500  
    model = RobertaAttentionForMultiLabel(Config, num_labels=num_labels)
    
    import os
    # 获取当前脚本所在绝对路径，确保加载模型时的路径正确
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "output_roberta_ASL_ATTENTION", "best_model_final", "pytorch_model.bin")
    
    # 如果你有训练好的权重，可以在这里加载（只有加载了真实权重，模型才会准确“看”到关键术语）
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(">> 成功加载训练好的真实模型权重！")
    else:
        print(">> 未找到模型权重，使用随机初始化状态！")
        
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # 2. 准备包含“卷积核”、“非易失性”等术语的测试文本
    text = "本发明公开了一种基于卷积神经网络的图像处理装置，系统通过特定的卷积核提取图像特征，并将关键参数存储在非易失性存储器中以实现高速推理。"
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    
    # 获取原始 token 列表用于显示在热力表上
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # 3. 运行模型，获取注意力权重
    with torch.no_grad():
        outputs = model(**inputs)
        # 形状: [1, seq_len]
        attn_weights = outputs["attn_weights"][0].cpu().numpy()

    # --- 数据处理 ---
    # 过滤掉 [CLS], [SEP] 和 [PAD] 等特殊字符，避免它们过大的初始权重压低真实词汇的显示比例
    filtered_tokens = []
    filtered_weights = []
    special_tokens = {"[CLS]", "[SEP]", "[PAD]"}
    
    for token, weight in zip(tokens, attn_weights):
        if token not in special_tokens:
            filtered_tokens.append(token)
            filtered_weights.append(weight)

    # --- 将字级权重合并为词级权重 (Word-level Aggregation) ---
    import jieba
    # 添加这几个核心技术术语到分词词典，确保它们作为完整的长词出现
    jieba.add_word("卷积核")
    jieba.add_word("非易失性")
    jieba.add_word("神经网络")
    jieba.add_word("存储器")
    words = list(jieba.cut(text))
    word_tokens = []
    word_weights = []
    
    char_idx = 0
    # 常用的停用词和标点，过滤掉它们可以减少柱子数量，放大核心词汇对比
    stop_words = {"本发明", "公开", "了", "一种", "基于", "的", "及", "系统", "通过", "特定", "并", "将", "在", "中", "以", "实现", "，", "。"}
    
    for w in words:
        w_len = len(w)
        # 将构成该词的所有字的权总和 (或求平均) 作为该词的权重
        # 此处使用求和
        if char_idx + w_len <= len(filtered_weights):
            w_weight = sum(filtered_weights[char_idx : char_idx + w_len])
            if w not in stop_words and len(w.strip()) > 0:
                word_tokens.append(w)
                word_weights.append(w_weight)
            char_idx += w_len

    # 4. 绘制更直观的柱状图 (Bar Plot) 替代一维热力带
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei'] # 支持中文字符
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 6))
    
    # 根据权重值生成颜色渐变 (权重高的颜色深)
    norm = plt.Normalize(min(word_weights), max(word_weights))
    colors = plt.cm.YlOrRd(norm(word_weights)) # 使用黄-橙-红渐变色

    # 绘制柱状图
    bars = plt.bar(word_tokens, word_weights, color=colors, edgecolor='gray', width=0.6)
    
    # 标出具体的数值 (调大字体)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.3f}', 
                 va='bottom', ha='center', fontsize=12, fontweight='bold', rotation=0)

    plt.title("Attention Pooling Token Weights (核心技术术语)", fontsize=18, pad=20)
    plt.xlabel("Tokens (Filtered)", fontsize=14)
    plt.ylabel("Attention Weight", fontsize=14)
    plt.xticks(rotation=30, ha='right', fontsize=14)
    
    # 添加网格线以便于阅读
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 5. 保存图表
    plt.savefig("attention_barplot.png", dpi=300)
    print("注意力权重分布图已保存为 attention_barplot.png")

if __name__ == "__main__":
    plot_attention_heatmap()
