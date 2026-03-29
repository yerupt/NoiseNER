# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertPreTrainedModel, BertModel


class BERT_BiLSTM_CRF(BertPreTrainedModel):

    def __init__(self, config, need_birnn=False, rnn_dim=128):
        super(BERT_BiLSTM_CRF, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size

        # ---------------------------------------------------------
        # 新增：Projector 和 Predictor 用于冷启动阶段的自监督损失计算
        # ---------------------------------------------------------
        self.projector = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim, bias=False)
        )

        self.predictor = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

        self.softmax = nn.Softmax(dim=-1)
        self.hidden2tag = nn.Linear(in_features=out_dim, out_features=config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.bert_layers = list(self.bert.encoder.layer.children())

    def forward(self, input_ids, tags=None, token_type_ids=None, attention_mask=None):
        """
        BERT_BiLSTM_CRF模型的正向传播函数

        :param input_ids:      torch.Size([batch_size,seq_len]), 代表输入实例的tensor张量
        :param tags:           torch.Size([batch_size,seq_len]), 真实标签序列
        :param token_type_ids: torch.Size([batch_size,seq_len]), 一个实例可以含有两个句子,相当于标记
        :param attention_mask: torch.Size([batch_size,seq_len]), 指定对哪些词进行self-Attention操作
        :return: loss, logits, sequence_output_dropped, proj, pred
        """
        if attention_mask is not None:
            attention_mask = attention_mask.byte()
        else:
            # 修正：使用 input_ids 生成掩码，避免 tags 为 None 时报错
            attention_mask = torch.ones_like(input_ids, dtype=torch.uint8)

        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        # Dropout 处理
        sequence_output_dropped = self.dropout(sequence_output)

        # ---------------------------------------------------------
        # 新增：通过 Projector 和 Predictor 提取特征，用于自监督对比
        # 将 3D 张量展平通过 BatchNorm1d，再还原形状
        # ---------------------------------------------------------
        flat_seq = sequence_output_dropped.view(-1, sequence_output_dropped.size(-1))
        proj = self.projector(flat_seq).view(sequence_output_dropped.size())
        pred = self.predictor(proj.view(-1, proj.size(-1))).view(proj.size())

        # 获取 CRF 发射矩阵
        emissions = self.hidden2tag(sequence_output_dropped)

        # 计算 CRF loss
        if tags is not None:
            loss = -1 * self.crf(emissions, tags, mask=attention_mask.byte())
        else:
            loss = None

        # 修正：将整个 emission 矩阵作为 logits 返回，方便与 CrossEntropy 结合及计算原型距离
        logits = emissions

        # 返回新增的特征表示用于 train.py 中的损失计算和 KDE 过滤
        return loss, logits, sequence_output_dropped, proj, pred

    def predict(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        模型预测
        :param input_ids:
        :param token_type_ids:
        :param attention_mask:
        :return: 经过 CRF 解码的最佳路径标签列表
        """
        if attention_mask is not None:
            attention_mask = attention_mask.byte()
        else:
            attention_mask = torch.ones_like(input_ids, dtype=torch.uint8)

        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)

        # 使用 CRF 维特比解码
        return self.crf.decode(emissions, attention_mask.byte())