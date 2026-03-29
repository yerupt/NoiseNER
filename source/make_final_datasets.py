import os


def build_datasets(pred_file_path, train_700_file_path, output_dir):
    """
    根据模型的预测结果，提取纯噪声样本 (Dataset 3)，并合成混合噪声样本 (Dataset 2)
    """
    print("⏳ 开始分析预测结果...")

    with open(pred_file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip().split('\n\n')

    dataset3_noise = []
    total_analyzed = len(content)

    for sent_block in content:
        lines = sent_block.split('\n')
        is_noisy = False
        original_sentence_lines = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                token = parts[0]
                true_label = parts[1]
                pred_label = parts[2]

                original_sentence_lines.append(f"{token} {true_label}")

                if true_label != pred_label:
                    is_noisy = True

        if is_noisy and original_sentence_lines:
            dataset3_noise.append('\n'.join(original_sentence_lines))

    os.makedirs(output_dir, exist_ok=True)
    dataset3_path = os.path.join(output_dir, 'Dataset3.txt')
    with open(dataset3_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(dataset3_noise) + '\n\n')

    print(f"✅ 从 {total_analyzed} 个候选样本中，成功提取了 {len(dataset3_noise)} 个纯噪声样本。")
    print(f"💾 Dataset 3 已保存至: {dataset3_path}")

    print("\n⏳ 开始合成 Dataset 2...")
    with open(train_700_file_path, 'r', encoding='utf-8') as f:
        train_700_content = f.read().strip().split('\n\n')

    dataset2_mixed = train_700_content + dataset3_noise

    dataset2_path = os.path.join(output_dir, 'Dataset2.txt')
    with open(dataset2_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(dataset2_mixed) + '\n\n')

    print(
        f"✅ 合成完毕！干净样本 ({len(train_700_content)}) + 噪声样本 ({len(dataset3_noise)}) = 混合样本 ({len(dataset2_mixed)})")
    print(f"💾 Dataset 2 已保存至: {dataset2_path}")
    print("\n🎉 数据工程全部完成，准备开启正式实验！")


if __name__ == '__main__':
    # ================= 配置路径区域 =================
    # 【请在此处填入你真实的本地路径】

    # 1. 刚才跑出来的预测结果
    PRED_RESULT_FILE = r'D:\Study\Project\NoiseNER\output\step1_base\20260319204439\token_labels_test.txt'

    # 2. 最初切出来的干净数据集
    TRAIN_700_FILE = r'D:\Study\Project\NoiseNER\data\step1_base\train.txt'

    # 3. 最终结果保存的位置
    OUTPUT_DIRECTORY = r'D:\Study\Project\NoiseNER\data\final_datasets'
    # ================================================

    if os.path.exists(PRED_RESULT_FILE) and os.path.exists(TRAIN_700_FILE):
        build_datasets(PRED_RESULT_FILE, TRAIN_700_FILE, OUTPUT_DIRECTORY)
    else:
        print("❌ 错误：找不到输入的 TXT 文件，请仔细检查上面的路径！")