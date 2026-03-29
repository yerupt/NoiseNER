# -*- coding: utf-8 -*-
import os
import pickle

import torch
from transformers import BertTokenizer

from config import Config


def get_entities_result(query, model_path=None):
    """
    进一步封装识别结果，最终结果格式如下:
    [
      {'type': 'APT', 'value': 'APT-C-05', 'begin': 0, 'end': 8},
      {'type': 'LOC', 'value': 'China', 'begin': 15, 'end': 20}
    ]
    :param query: 查询问句
    :param model_path: 模型保存路径，默认取 Config 中的 output_path
    :return: 实体列表
    """
    if model_path is None:
        config = Config()
        model_path = config.output_path

    # 检查模型是否存在
    if not os.path.exists(os.path.join(model_path, "ner_model.ckpt")):
        print(f"找不到模型文件，请检查路径: {model_path}")
        return []

    sentence_list, predict_labels = predict(query, model_path)

    if len(predict_labels) == 0:
        print("句子: {0}\t实体识别结果为空".format(query))
        return []

    entities = []
    if len(sentence_list) == len(predict_labels):
        # 替换为兼容 BIESO 格式的新版解析器
        result = _bieso_data_handler(sentence_list, predict_labels)
        if len(result) != 0:
            end = 0
            prefix_len = 0

            for word, label in result:
                sen = query.lower()[end:]
                # 寻找实体在原句中的起始和结束索引
                begin = sen.find(word.lower()) + prefix_len
                end = begin + len(word)
                prefix_len = end
                if begin != -1:
                    ent = dict(value=query[begin:end], type=label, begin=begin, end=end)
                    entities.append(ent)
    return entities


def predict(sentence, model_path):
    """
    模型预测 (根据论文 3.5 节，测试阶段仅需选取验证集表现最优的单模型进行推理)
    :param sentence: 待预测字符串
    :param model_path: 模型路径
    :return: 字列表，预测标签列表
    """
    config = Config()
    max_seq_length = config.max_seq_length
    if len(sentence) > max_seq_length:
        sentence = sentence[:max_seq_length - 2]

    # 加载 label 映射字典
    with open(os.path.join(model_path, "label2id.pkl"), "rb") as f:
        label2id = pickle.load(f)
    id2label = {value: key for key, value in label2id.items()}

    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=config.do_lower_case)

    # 细粒度分词
    tokens = []
    for word in list(sentence):
        tokenized_word = tokenizer.tokenize(word)
        if not tokenized_word:
            tokenized_word = ['[UNK]']
        tokens.extend(tokenized_word)

    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]

    # 获取句子的input_ids、token_type_ids、attention_mask
    result = tokenizer.encode_plus(tokens, add_special_tokens=True, max_length=max_seq_length,
                                   padding="max_length")
    input_ids, token_type_ids, attention_mask = result["input_ids"], result["token_type_ids"], result["attention_mask"]

    # 动态设备选择
    device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu')

    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    token_type_ids = torch.tensor([token_type_ids], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)

    # 加载已保存的最佳模型 ner_model.ckpt
    model = torch.load(os.path.join(model_path, "ner_model.ckpt"), map_location=device)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()

    # 模型预测，不需要反向传播
    with torch.no_grad():
        predict_val = model.predict(input_ids, token_type_ids, attention_mask)

    predict_labels = []
    # CRF decode 返回的结果去掉了 padding，但是包含了首尾的 [CLS] 和 [SEP]
    # 我们只截取有效长度的标签
    valid_len = len(tokens)
    for i, label in enumerate(predict_val[0]):
        if i == 0: continue  # 跳过 [CLS]
        if i > valid_len: break  # 到达 [SEP] 停止
        predict_labels.append(id2label[label])

    return list(sentence), predict_labels


def _bieso_data_handler(sentence, predict_label):
    """
    处理 BIESO/BIO 开头的标签信息 (适配论文网络安全数据集 APTNER)
    标签说明：
    B: Begin 实体开始
    I: Inside 实体内部
    E: End 实体结束
    S: Single 独立成实体
    O: Outside 非实体
    """
    entities = []
    word = ""
    current_type = ""

    for i in range(len(sentence)):
        char = sentence[i]
        label = predict_label[i]

        if label == 'O':
            if word != "":
                entities.append([word, current_type])
                word = ""
                current_type = ""

        elif label.startswith('B-'):
            if word != "":
                entities.append([word, current_type])
            word = char
            current_type = label[2:]

        elif label.startswith('I-') or label.startswith('M-'):
            if current_type == label[2:]:
                word += char
            else:
                if word != "":
                    entities.append([word, current_type])
                word = char
                current_type = label[2:]

        elif label.startswith('E-'):
            if current_type == label[2:]:
                word += char
                entities.append([word, current_type])
                word = ""
                current_type = ""
            else:
                if word != "":
                    entities.append([word, current_type])
                # 即使类型不匹配，为了容错，将 E 当作独立实体或截断记录
                entities.append([char, label[2:]])
                word = ""
                current_type = ""

        elif label.startswith('S-'):
            if word != "":
                entities.append([word, current_type])
                word = ""
            entities.append([char, label[2:]])
            current_type = ""

    # 收尾：若最后词没有遇到 O 或 E 闭合，强制闭合并保存
    if word != "":
        entities.append([word, current_type])

    return entities


if __name__ == '__main__':
    # 测试网络安全领域的句子，APTNER 类别一般包括: TIME, IDTY, APT, MAL, TOOL
    sent = "The threat converts the command result text encoding from Cyrillic to UTF-16"

    config = Config()
    # 如果没训练出模型，可以用一个已有的目录手动测试，例如：
    # test_path = os.path.join(config.base_path, "output", "20240828113543")
    # entities = get_entities_result(sent, model_path=test_path)

    entities = get_entities_result(sent)
    print("识别结果:", entities)