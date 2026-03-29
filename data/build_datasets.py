import random

# 论文中指定的 5 个目标类别
TARGET_CATEGORIES = ['TIME', 'IDTY', 'APT', 'MAL', 'TOOL']


def load_data(file_list):
    """读取多个 txt 文件，并按空行拆分成句子列表"""
    all_sentences = []
    for file_path in file_list:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 按照双换行符切分出独立的句子
                sentences = f.read().strip().split('\n\n')
                all_sentences.extend(sentences)
        except FileNotFoundError:
            print(f"警告：未找到文件 {file_path}")
    return all_sentences


def get_categories_in_sentence(sentence):
    """检测一个句子中包含了哪些目标实体类别"""
    found_cats = set()
    for line in sentence.split('\n'):
        parts = line.strip().split()
        if len(parts) >= 2:
            label = parts[-1]
            for cat in TARGET_CATEGORIES:
                if cat in label:
                    found_cats.add(cat)
    return found_cats


def main():
    print("开始加载原始数据...")
    # 把你现有的三个文件读进来作为一个大池子
    pool = load_data(['train.txt', 'test.txt', 'eval.txt'])
    print(f"总计读取到 {len(pool)} 个句子。")

    # 用于存放每个类别的 1000 个样本
    category_samples = {cat: [] for cat in TARGET_CATEGORIES}

    # Dataset 1：不属于这 5x1000 个样本的剩余数据
    dataset1_remaining = []

    # 1. 遍历数据池，为每个类别挑选 1000 条样本
    for sent in pool:
        cats_in_sent = get_category_counts(sent)
        assigned = False

        # 优先把句子分配给还没满 1000 条的类别
        for cat in cats_in_sent:
            if len(category_samples[cat]) < 1000:
                category_samples[cat].append(sent)
                assigned = True
                break  # 一个句子只算作一个主要类别的代表，防止重复计算

        # 如果这个句子没有被分配到任何 1000 条的池子里，就归入 Dataset 1
        if not assigned:
            dataset1_remaining.append(sent)

    print(f"Dataset 1 (最终测试集) 样本数: {len(dataset1_remaining)}")

    # 2. 从每个类别的 1000 条中，拆分 700(训练) 和 300(预测候选)
    bilstm_train_700 = []
    noise_candidates_300 = []

    for cat, sents in category_samples.items():
        print(f"类别 {cat} 收集到样本数: {len(sents)}")
        # 打乱顺序保证随机性
        random.seed(42)
        random.shuffle(sents)

        # 前 700 条用于训练 BiLSTM
        bilstm_train_700.extend(sents[:700])
        # 后 300 条用于预测和筛选噪声
        noise_candidates_300.extend(sents[700:])

    # 3. 将结果写回 TXT 文件 (依然保持 CoNLL 原生格式)
    def save_to_txt(filename, data_list):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(data_list) + '\n\n')

    save_to_txt('Dataset1.txt', dataset1_remaining)
    save_to_txt('step1_base/bilstm_train_700.txt', bilstm_train_700)
    save_to_txt('step1_base/noise_candidates_300.txt', noise_candidates_300)

    print("\n✅ 数据划分完成！")
    print("生成文件: Dataset1.txt (用于最终测试)")
    print("生成文件: bilstm_train_700.txt (用于训练基础 BiLSTM)")
    print("生成文件: noise_candidates_300.txt (给 BiLSTM 预测以提取纯噪声)")


if __name__ == '__main__':
    # 补充定义 get_category_counts 以防止报错
    def get_category_counts(sentence):
        return get_categories_in_sentence(sentence)


    main()