# config.py
import torch

class Config:
    MODEL_NAME = "/root/autodl-tmp/chinese-roberta-wwm-ext"
    
    # [新增] 模型头部类型选择："CNN" 或 "BASELINE" 或 "ATTENTION"(BASELINE+注意力机制) 或 "CNN+ATTENTION"
    MODEL_HEAD = "CNN+ATTENTION"

    # 是否启用层级 sibling penalty (ASL 变体). 用于消融实验。
    USE_SIBLING_PENALTY = True

    # 500类×40主实验
    # TRAIN_CSV = "data/g06f_train.csv"
    # VAL_CSV = "data/g06f_val.csv"
    # TEST_CSV = "data/g06f_test.csv"
    # LABEL_JSON = "data/g06f_selected_labels.json"
    # OUTPUT_DIR = "./output_500x40"

    # 10类×2000诊断实验
    # TRAIN_CSV = "data/g06f_top10_2000_train.csv"
    # VAL_CSV   = "data/g06f_top10_2000_val.csv"
    # LABEL_JSON = "data/g06f_top10_labels.json"
    # OUTPUT_DIR = "./output_top10"

    # 500类×80主实验
    # TRAIN_CSV = "/root/autodl-tmp/g06f_500x80_train.csv"
    # VAL_CSV   = "/root/autodl-tmp/g06f_500x80_val.csv"
    # LABEL_JSON = "/root/autodl-tmp/g06f_500x80_labels.json"
    # OUTPUT_DIR = "./output_g06f_500x80"
    # TEST_CSV = "/root/autodl-tmp/g06f_500x80_test.csv"

    # 500类×80_fixed_实验
    # TRAIN_CSV = "/root/autodl-tmp/fixed_v2_120/train_120.csv"
    # VAL_CSV   = "/root/autodl-tmp/fixed_v2_120/val_fixed.csv"
    # LABEL_JSON = "/root/autodl-tmp/fixed_v2_120/selected_labels.json"
    # OUTPUT_DIR = "./output_500x120_fixed_120"
    # TEST_CSV = "/root/autodl-tmp/fixed_v2_120/test_fixed.csv"

    # 500类×120_按类保底构造集
    # TRAIN_CSV = "/root/autodl-tmp/g06f_oldnew_cov_500x120/g06f_500x120_train.csv"
    # VAL_CSV = "/root/autodl-tmp/g06f_oldnew_cov_500x120/g06f_500x120_val.csv"
    # TEST_CSV = "/root/autodl-tmp/g06f_oldnew_cov_500x120/g06f_500x120_test.csv"
    # LABEL_JSON = "/root/autodl-tmp/g06f_oldnew_cov_500x120/g06f_500x120_labels.json"
    # OUTPUT_DIR = "./output_roberta_cnn"

    # 自然测试集+ASL损失函数
    TRAIN_CSV = "/root/autodl-tmp/fixed_v2_120/train_120.csv"
    VAL_CSV = "/root/autodl-tmp/fixed_v2_120/val_fixed.csv"
    TEST_CSV = "/root/autodl-tmp/fixed_v2_120/test_fixed.csv"
    LABEL_JSON = "/root/autodl-tmp/fixed_v2_120/selected_labels.json"
    OUTPUT_DIR = "./output_roberta_ASL_ATTENTION_penalty"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    MAX_LEN = 384
    BATCH_SIZE = 32
    LR = 2e-5
    EPOCHS = 5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    SEED = 42
    
    NUM_LABELS = 500
    METRIC_FOR_BEST_MODEL = "hr@3"

    CNN_KERNEL_SIZES = [3, 4, 5]
    CNN_NUM_FILTERS = 256
    DROPOUT = 0.2

    # ASL 的超参数
    GAMMA_NEG = 5
    GAMMA_POS = 0
    CLIP = 0.1

    # 层级惩罚的 ASL 的超参数
    SIBLING_DISCOUNT = 0.9
