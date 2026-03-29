import os
import random


def split_dataset(input_file, eval_file, test_file, split_ratio=0.5, seed=42):
    print(f"⏳ 开始读取并切分数据集: {input_file}")

    # 设置随机种子，保证你的实验绝对可复现（非常符合学术规范）
    random.seed(seed)

    # 1. 读取原始 CoNLL 数据
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # 按双换行符切割成独立的句子块
    sentences = content.split('\n\n')
    # 过滤掉可能的空行干扰
    sentences = [s for s in sentences if s.strip()]

    total_samples = len(sentences)
    print(f"✅ 成功读取 {total_samples} 个有效句子。")

    # 2. 全局打乱，保证验证集和测试集的类别分布均匀
    random.shuffle(sentences)

    # 3. 按比例切分 (默认 1:1)
    split_idx = int(total_samples * split_ratio)
    eval_sentences = sentences[:split_idx]
    test_sentences = sentences[split_idx:]

    # 4. 写入验证集 (Eval)
    with open(eval_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(eval_sentences) + '\n\n')

    # 5. 写入测试集 (Test)
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(test_sentences) + '\n\n')

    print(f"🎉 完美切分完成！")
    print(f"💾 验证集 (Eval) 已保存至: {eval_file} (共 {len(eval_sentences)} 条)")
    print(f"💾 测试集 (Test) 已保存至: {test_file} (共 {len(test_sentences)} 条)")


if __name__ == '__main__':
    # ================= 配置路径区域 =================
    # 1. 你的那个 5263 条干净数据的 Dataset1.txt 路径
    INPUT_DATASET = r'D:\Study\Project\NoiseNER\data\Dataset1.txt'

    # 2. 输出文件夹路径 (建议统一放在 final_datasets 里方便管理)
    OUTPUT_DIR = r'D:\Study\Project\NoiseNER\data\final_datasets'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    OUTPUT_EVAL = os.path.join(OUTPUT_DIR, 'eval.txt')
    OUTPUT_TEST = os.path.join(OUTPUT_DIR, 'test.txt')
    # ================================================

    if os.path.exists(INPUT_DATASET):
        split_dataset(INPUT_DATASET, OUTPUT_EVAL, OUTPUT_TEST)
    else:
        print("❌ 找不到输入文件，请检查 INPUT_DATASET 的路径！")