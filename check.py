import os


def check_subset(full_dataset_path, subset_path, name):
    # 读取总数据集
    with open(full_dataset_path, 'r', encoding='utf-8') as f:
        # 使用 set (集合) 来存储，查找速度快，且忽略顺序
        full_data = set(f.read().splitlines())

    # 读取子数据集
    with open(subset_path, 'r', encoding='utf-8') as f:
        subset_data = set(f.read().splitlines())

    # 检查是否是子集
    is_subset = subset_data.issubset(full_data)

    if is_subset:
        print(f"✅ 验证成功: {name} 是 dataset2.txt 的一部分。")
    else:
        # 计算有多少行不在总集中
        diff = subset_data - full_data
        print(f"❌ 验证失败: {name} 中有 {len(diff)} 行数据不在 dataset2.txt 中。")


import os

# ... (check_subset 函数部分保持不变) ...

if __name__ == "__main__":
    # 请将下面的路径替换为你电脑上文件的真实完整路径
    # 示例： D:/Study/Project/NoiseNER/dataset2/dataset2.txt

    full_path = r"D:\Study\Project\NoiseNER\data\dataset2.txt"
    train_path = r"data/bz/train.txt"
    eval_path = r"data/bz/eval.txt"

    # 验证 train.txt
    if os.path.exists(full_path) and os.path.exists(train_path):
        check_subset(full_path, train_path, "train.txt")
    else:
        print(f"❌ 找不到文件！请检查路径是否正确。\n试图查找: {full_path} 和 {train_path}")

    # 验证 eval.txt
    if os.path.exists(full_path) and os.path.exists(eval_path):
        check_subset(full_path, eval_path, "eval.txt")