# -*- coding: utf-8 -*-
import os
import random
import pickle
import numpy as np
import torch

def set_seed(seed=42):
    """
    固定所有随机种子以确保学术论文实验的完全可复现性
    :param seed: 随机种子，默认值为42
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置种子
    # 保证每次返回的卷积算法是确定的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_file(fp: str, sep: str = None, name_tuple=None):
    """
    读取文件；
    若sep为None，按行读取，返回文件内容列表，格式为:[xxx,xxx,xxx,...]
    若不为None，按行读取分隔，返回文件内容列表，格式为: [[xxx,xxx],[xxx,xxx],...]
    :param fp:
    :param sep:
    :param name_tuple:
    :return:
    """
    with open(fp, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if sep:
            if name_tuple:
                return map(name_tuple._make, [line.strip().split(sep) for line in lines])
            else:
                return [line.strip().split(sep) for line in lines]
        else:
            return lines

def load_pkl(fp):
    """
    加载pkl文件
    :param fp:
    :return:
    """
    with open(fp, 'rb') as f:
        data = pickle.load(f)
        return data

def save_pkl(data, fp):
    """
    保存pkl文件，数据序列化
    :param data:
    :param fp:
    :return:
    """
    with open(fp, 'wb') as f:
        pickle.dump(data, f)