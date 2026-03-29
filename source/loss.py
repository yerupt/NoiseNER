# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def regression_loss(x, y):
    # x, y are in shape (N, C)
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return 2 - 2 * (x * y).sum(dim=-1)


def entropy(p):
    return Categorical(probs=p).entropy()


def entropy_loss(logits, reduction='mean'):
    losses = entropy(F.softmax(logits, dim=1))  # (N)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def cross_entropy(logits, labels, reduction='mean'):
    """
    计算软标签(分布)的交叉熵损失
    :param logits: shape: (N, C) 未经过softmax的模型输出
    :param labels: shape: (N, C) 目标概率分布或one-hot标签
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'

    log_logits = F.log_softmax(logits, dim=1)
    losses = -torch.sum(log_logits * labels, dim=1)  # (N)

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def label_smoothing_cross_entropy(logits, labels, epsilon=0.1, reduction='none'):
    N = logits.size(0)
    C = logits.size(1)
    smoothed_label = torch.full(size=(N, C), fill_value=epsilon / (C - 1))
    smoothed_label.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1).cpu(), value=1 - epsilon)
    if logits.is_cuda:
        smoothed_label = smoothed_label.cuda()
    return cross_entropy(logits, smoothed_label, reduction)


# ==========================================
# 新增模块：论文特有的多约束损失函数
# ==========================================

def semantic_similarity_loss(x_mix, x_unclean, x_clean, y_mix, y_unclean, y_clean, eps=1e-8):
    """
    基于语义相似度的标签序列分布学习损失 (对应论文 Eq. 20)
    目标1: 混合标签 y_mix 尽量与干净标签 y_clean 相似
    目标2: 混合表征 x_mix 尽量与噪声样本表征 x_unclean 相似 (以学习更多样化的特征)
    """
    # 计算 y 的余弦相似度 (为防止分母为0，加入 eps 平滑项)
    sim_y_num = F.cosine_similarity(y_mix, y_unclean, dim=-1)
    sim_y_den = F.cosine_similarity(y_mix, y_clean, dim=-1)
    # 将余弦相似度映射到正数区间 [0, 2] 避免出现负数导致优化异常
    sim_y_num = sim_y_num + 1.0 
    sim_y_den = sim_y_den + 1.0
    l_simy = (sim_y_num + eps) / (sim_y_den + eps)

    # 计算 x 的余弦相似度
    sim_x_num = F.cosine_similarity(x_mix, x_clean, dim=-1)
    sim_x_den = F.cosine_similarity(x_mix, x_unclean, dim=-1)
    sim_x_num = sim_x_num + 1.0
    sim_x_den = sim_x_den + 1.0
    l_simx = (sim_x_num + eps) / (sim_x_den + eps)

    # 总体相似度偏差损失
    l_sim = l_simy + l_simx
    return l_sim.mean()


def consistency_loss(logits, target_probs, reduction='batchmean'):
    """
    改进版一致性损失：使用 KL 散度实现分布对齐
    logits: 当前模型的原始输出 (Raw Logits)
    target_probs: 另一个模型产生的概率分布 (Softmax Probs, 已 detach)
    """
    # 将当前模型的 logits 转为 Log-Probability 空间
    log_probs = F.log_softmax(logits, dim=-1)

    # 计算 KL 散度：KL(Target || Current)
    # PyTorch 的 KLDiv 默认输入顺序是 (input_log_probs, target_probs)
    return F.kl_div(log_probs, target_probs, reduction=reduction)


class SmoothingLabelCrossEntropyLoss(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self._epsilon = epsilon
        self._reduction = reduction

    def forward(self, logits, labels):
        return label_smoothing_cross_entropy(logits, labels, self._epsilon, self._reduction)


class ScatteredCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self._reduction = reduction

    def forward(self, logits, labels):
        return cross_entropy(logits, labels, self._reduction)