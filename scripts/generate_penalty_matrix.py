import json
import torch
import re

def parse_ipc_g06f(label_str):
    """
    解析 G06F 的标签结构
    假设你的标签格式类似于 "G06F 17/21" 或 "G06F17/21" 或 "17/21"
    提取出 main_group (17) 和 sub_group (21)
    """
    # 使用正则匹配数字部分，例如匹配出 17 和 21
    match = re.search(r'(\d+)/(\d+)', str(label_str))
    if match:
        main_group = match.group(1)
        sub_group = match.group(2)
        return main_group, sub_group
    else:
        # 如果格式不标准，退化处理
        return label_str, "00"

def create_penalty_matrix(label_json_path, save_path="penalty_matrix.pt"):
    with open(label_json_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    
    num_labels = len(labels)
    penalty_matrix = torch.zeros((num_labels, num_labels), dtype=torch.float32)

    for i in range(num_labels):
        main_i, sub_i = parse_ipc_g06f(labels[i])
        for j in range(num_labels):
            if i == j:
                penalty_matrix[i][j] = 0.0
            else:
                main_j, sub_j = parse_ipc_g06f(labels[j])
                if main_i == main_j:
                    # 同一个主组，惩罚较小
                    penalty_matrix[i][j] = 1.0
                else:
                    # 跨主组，惩罚加大
                    penalty_matrix[i][j] = 2.0

    # 归一化矩阵（可选，防止 loss 过大），这里我们将最大距离缩放到 1.0
    penalty_matrix = penalty_matrix / penalty_matrix.max()
    
    torch.save(penalty_matrix, save_path)
    print(f"惩罚矩阵已生成并保存至 {save_path}，形状: {penalty_matrix.shape}")

if __name__ == "__main__":
    # 替换为你 Config 中的 LABEL_JSON 路径
    create_penalty_matrix("/root/autodl-tmp/fixed_v2_120/selected_labels.json", "penalty_matrix.pt")
