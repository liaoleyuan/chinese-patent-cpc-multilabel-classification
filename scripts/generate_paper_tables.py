import os
import json
import csv

# ================= 配置区 =================
# 项目根目录 (由于你已经在 PROJECT_2 下，直接用当前目录即可)
ROOT_DIR = "."
METRICS_FILE = "final_test_metrics.json"

# 定义消融实验(Ablation Study)的模型演进路径
# Key 为论文中展示的名称，Value 为对应的实际文件夹名称
ABLATION_MAPPING = {
    "Baseline (RoBERTa)": "output_500x120_fixed_120", 
    "+ ASL Loss": "output_roberta_ASL",
    "+ Attention Pooling": "output_roberta_ASL_ATTENTION",
    "+ Hierarchy Penalty": "output_roberta_ASL_penalty",
    "Ours (Full Model)": "output_roberta_ASL_ATTENTION_penalty_CNN_1e-5_1e-4"
}

# ==========================================

def load_metrics(folder_path):
    """尝试从指定文件夹加载评估指标"""
    file_path = os.path.join(folder_path, METRICS_FILE)
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[警告] 无法读取 {file_path}: {e}")
        return None

def generate_global_csv():
    """扫描所有包含 output_ 的文件夹，导出全量实验数据"""
    print(">>> 正在提取所有实验数据并生成 all_experiments_results.csv ...")
    all_data = []
    
    for folder_name in os.listdir(ROOT_DIR):
        if folder_name.startswith("output_") and os.path.isdir(os.path.join(ROOT_DIR, folder_name)):
            metrics = load_metrics(os.path.join(ROOT_DIR, folder_name))
            if metrics:
                # 扁平化字典，加入文件夹名称作为实验标识
                row = {"Experiment": folder_name}
                row.update(metrics)
                all_data.append(row)
                
    if not all_data:
        print("[提示] 没有找到任何包含 final_test_metrics.json 的 output 文件夹。")
        return

    # 获取所有可能出现的表头
    fieldnames = set()
    for data in all_data:
        fieldnames.update(data.keys())
    # 确保 Experiment 在第一列
    fieldnames = ["Experiment"] + sorted([f for f in fieldnames if f != "Experiment"])

    with open("all_experiments_results.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)
    print(">>> 成功生成全量数据表: all_experiments_results.csv\n")

def generate_ablation_table():
    """生成用于论文的消融实验 Markdown 和 LaTeX 表格"""
    print(">>> 正在生成核心消融实验 (Ablation Study) 表格 ...\n")
    
    # 准备表头
    headers = ["Model", "HitRate@1", "HitRate@3", "HitRate@5", "HitRate@10", "Micro-F1"]
    rows = []
    
    for display_name, folder_name in ABLATION_MAPPING.items():
        metrics = load_metrics(os.path.join(ROOT_DIR, folder_name))
        if metrics:
            # 这里的键名需要根据你 json 文件里实际的键名进行修改
            # 比如可能是 'eval_hr@1', 'eval_f1' 等
            hr1 = metrics.get('hr@1', metrics.get('eval_hr@1', '-'))
            hr3 = metrics.get('hr@3', metrics.get('eval_hr@3', '-'))
            hr5 = metrics.get('hr@5', metrics.get('eval_hr@5', '-'))
            hr10 = metrics.get('hr@10', metrics.get('eval_hr@10', '-'))
            f1 = metrics.get('f1', metrics.get('eval_f1', metrics.get('micro_f1', '-')))
            
            # 如果是浮点数，保留四位小数或转为百分比
            def format_metric(m):
                if isinstance(m, (int, float)):
                    return f"{m * 100:.2f}" # 转为带两位小数的百分比格式
                return str(m)
                
            rows.append([
                display_name,
                format_metric(hr1),
                format_metric(hr3),
                format_metric(hr5),
                format_metric(hr10),
                format_metric(f1)
            ])
        else:
            rows.append([display_name, "N/A", "N/A", "N/A", "N/A", "N/A"])

    # 1. 打印 Markdown 表格
    print("【Markdown 格式表格】(可直接复制到语雀/Notion/Typora)")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print("\n")

    # 2. 打印 LaTeX 表格
    print("【LaTeX 格式表格】(可直接复制到 Overleaf)")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{模型核心模块消融实验结果（\\%）}")
    print("\\label{tab:ablation}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print(" & ".join(headers) + " \\\\")
    print("\\midrule")
    for row in rows:
        print(" & ".join(row) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}\n")

if __name__ == "__main__":
    generate_global_csv()
    generate_ablation_table()
