# -*- coding: utf-8 -*-
import os
import json
import logging
import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, text, label):
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids, ori_tokens):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids
        self.ori_tokens = ori_tokens


class NerProcessor(object):
    def clean_output(self, config):
        """清理历史残余缓存文件，防止旧数据干扰"""
        if os.path.exists(config.output_path):
            for file in os.listdir(config.output_path):
                if file.endswith(".json") or file.endswith(".cache"):
                    os.remove(os.path.join(config.output_path, file))
        else:
            os.makedirs(config.output_path, exist_ok=True)

    def read_data(self, input_file):
        """读取 CoNLL 格式的 TXT 数据"""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                if not contends:
                    if len(words) > 0:
                        lines.append((words, labels))
                        words = []
                        labels = []
                    continue
                parts = contends.split()
                if len(parts) >= 2:
                    words.append(parts[0])
                    labels.append(parts[-1])
            if len(words) > 0:
                lines.append((words, labels))
            return lines

    def get_labels(self, config=None):
        """
        [终极升级版] 全局扫描 + 强制兜底
        彻底解决拆分数据集导致某些类别标签遗漏（如 B-SHA2）的问题
        """
        labels = set(["O"])

        # 1. 动态扫描所有可能存在的数据文件
        if config is not None:
            # 兼容各种路径配置方式
            files_to_scan = []
            if hasattr(config, 'train_file'):
                files_to_scan.append(config.train_file)
            elif hasattr(config, 'data_dir'):
                files_to_scan.append(os.path.join(config.data_dir, 'train.txt'))

            if hasattr(config, 'eval_file'):
                files_to_scan.append(config.eval_file)
            elif hasattr(config, 'data_dir'):
                files_to_scan.append(os.path.join(config.data_dir, 'eval.txt'))

            if hasattr(config, 'test_file'):
                files_to_scan.append(config.test_file)
            elif hasattr(config, 'data_dir'):
                files_to_scan.append(os.path.join(config.data_dir, 'test.txt'))

            for file_path in set(files_to_scan):
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            contends = line.strip()
                            if contends and not contends.startswith("-DOCSTART-"):
                                parts = contends.split()
                                if len(parts) >= 2:
                                    labels.add(parts[-1])

        # 2. APTNER 数据集 BIESO 全量实体强制兜底
        base_types = [
            'ACT', 'APT', 'DOM', 'EMAIL', 'ENCR', 'FILE', 'IDTY', 'IP', 'LOC',
            'MAL', 'MD5', 'OS', 'PROT', 'SECTEAM', 'SHA1', 'SHA2', 'TIME',
            'TOOL', 'URL', 'VULID', 'VULNAME', 'APT\xa0as', 'APT\xa0is', 'S-SECTEAM'
        ]
        for prefix in ['B-', 'I-', 'E-', 'S-']:
            for t in base_types:
                labels.add(prefix + t)

        labels.add('PROT')  # 处理官方数据集中的特殊裸标签

        label_list = list(labels)
        label_list.sort()
        return label_list

    def get_label2id_id2label(self, output_path, label_list):
        label2id_path = os.path.join(output_path, "label2id.json")
        id2label_path = os.path.join(output_path, "id2label.json")

        # 强制每次覆盖生成最新的映射表
        label2id = {label: i for i, label in enumerate(label_list)}
        id2label = {i: label for i, label in enumerate(label_list)}

        with open(label2id_path, "w", encoding="utf-8") as f:
            json.dump(label2id, f, ensure_ascii=False, indent=4)
        with open(id2label_path, "w", encoding="utf-8") as f:
            json.dump(id2label, f, ensure_ascii=False, indent=4)

        return label2id, id2label

    def get_dataset(self, config, tokenizer, mode="train"):
        file_path = ""
        if mode == "train":
            file_path = config.train_file if hasattr(config, 'train_file') else os.path.join(config.data_dir,
                                                                                             "train.txt")
        elif mode == "eval":
            file_path = config.eval_file if hasattr(config, 'eval_file') else os.path.join(config.data_dir, "eval.txt")
        elif mode == "test":
            file_path = config.test_file if hasattr(config, 'test_file') else os.path.join(config.data_dir, "test.txt")

        lines = self.read_data(file_path)
        examples = []
        for i, (words, labels) in enumerate(lines):
            examples.append(InputExample(guid=f"{mode}-{i}", text=" ".join(words), label=" ".join(labels)))

        features = self.convert_examples_to_features(config, examples, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask, all_label_ids)
        return examples, features, dataset

    def convert_examples_to_features(self, config, examples, tokenizer):
        label_list = self.get_labels(config)
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        max_seq_length = config.max_seq_length

        for (ex_index, example) in enumerate(examples):
            textlist = example.text.split(" ")
            labellist = example.label.split(" ")
            tokens = []
            labels = []
            ori_tokens = []

            for i, word in enumerate(textlist):
                token = tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = labellist[i]
                for m in range(len(token)):
                    if m == 0:
                        labels.append(label_1)
                    else:
                        labels.append(label_1)
                ori_tokens.append(word)

            if len(tokens) >= max_seq_length - 2:
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]

            ntokens = []
            segment_ids = []
            label_ids = []
            ntokens.append("[CLS]")
            segment_ids.append(0)
            label_ids.append(label_map["O"])

            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
                # [无敌防爆机制]：如果真的遇到任何诡异标签，直接容错转为 'O'，绝不崩溃！
                label_ids.append(label_map.get(labels[i], label_map["O"]))

            ntokens.append("[SEP]")
            segment_ids.append(0)
            label_ids.append(label_map["O"])

            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                label_ids.append(0)

            if ex_index < 3:
                logger.info("*** 样例 %d 统计 ***", ex_index)
                logger.info("有效 token 数: %d", len(ntokens))

            features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=input_mask,
                              token_type_ids=segment_ids,
                              label_ids=label_ids,
                              ori_tokens=ori_tokens)
            )
        return features