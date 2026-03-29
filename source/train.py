# -*- coding: utf-8 -*-
import logging
import os
import random

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from tqdm import tqdm, trange

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, BertConfig, BertTokenizer
import numpy as np
import conlleval as conlleval
from config import Config
from models import BERT_BiLSTM_CRF
from data_processor import NerProcessor

# 引入完整的多约束损失函数
from loss import cross_entropy, entropy_loss, regression_loss, semantic_similarity_loss, consistency_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """固定所有随机种子以确保实验可复现性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# 新增模块：标签分布数据追踪器 (用于论文统计)
# ==========================================
class LabelStatsTracker:
    def __init__(self, id2label):
        self.id2label = id2label
        self.original_counts = {}
        self.relabeled_counts = {}
        self.total_tokens = 0

    def update(self, original_labels, b_out_mask, active_mask):
        active_flat = active_mask.view(-1) == 1
        orig_flat = original_labels.view(-1)[active_flat]
        bout_flat = b_out_mask.view(-1)[active_flat]

        self.total_tokens += orig_flat.size(0)

        # 1. 统计原始分布
        orig_unique, orig_counts = torch.unique(orig_flat, return_counts=True)
        for u, c in zip(orig_unique, orig_counts):
            label = self.id2label[u.item()]
            self.original_counts[label] = self.original_counts.get(label, 0) + c.item()

        # 2. 统计重标注后分布 (被判定为 b_out_mask 的，转换为 'O' 标签)
        relabeled_flat = orig_flat.clone()
        # 假设 'O' 标签对应的 ID 为 0
        relabeled_flat[bout_flat] = 0

        relab_unique, relab_counts = torch.unique(relabeled_flat, return_counts=True)
        for u, c in zip(relab_unique, relab_counts):
            label = self.id2label[u.item()]
            self.relabeled_counts[label] = self.relabeled_counts.get(label, 0) + c.item()

    def print_stats(self):
        if self.total_tokens == 0:
            return

        print("\n" + "=" * 60)
        print("📊 训练集 Token 级标签分布统计 (Before vs After Relabeling)")
        print("=" * 60)
        print(f"Total Valid Tokens: {self.total_tokens}")

        orig_o_count = self.original_counts.get('O', 0)
        relab_o_count = self.relabeled_counts.get('O', 0)

        print("\n[原始标签分布 (Before)]")
        print(
            f"  -> 'O' 标签占比: {orig_o_count} / {self.total_tokens} ({(orig_o_count / self.total_tokens) * 100:.2f}%)")
        for label, count in sorted(self.original_counts.items(), key=lambda x: x[0]):
            if label != 'O':
                print(f"  {label}: {count}")

        print("\n[重标注/剔除 OOD 噪声后 (After)]")
        print(
            f"  -> 'O' 标签占比: {relab_o_count} / {self.total_tokens} ({(relab_o_count / self.total_tokens) * 100:.2f}%)")
        print(f"  -> 新增 'O' 标签数量 (被转化为背景的 OOD 噪声): {relab_o_count - orig_o_count}")
        for label, count in sorted(self.relabeled_counts.items(), key=lambda x: x[0]):
            if label != 'O':
                print(f"  {label}: {count}")
        print("=" * 60 + "\n")


# ==========================================
# 模块：自监督对比损失与细粒度样本划分
# ==========================================

def self_supervised_loss(pred_1, proj_2, pred_2, proj_1, active_mask):
    active_loss = active_mask.view(-1)

    if active_loss.sum() == 0:
        return torch.tensor(0.0, device=pred_1.device), torch.tensor(0.0, device=pred_2.device)

    # 1. 提取有效部分
    p1_active = pred_1.view(-1, pred_1.size(-1))[active_loss]
    z2_active = proj_2.view(-1, proj_2.size(-1))[active_loss].detach()
    p2_active = pred_2.view(-1, pred_2.size(-1))[active_loss]
    z1_active = proj_1.view(-1, proj_1.size(-1))[active_loss].detach()

    # 2. 增加防爆 EPS (eps=1e-8) 强制稳定归一化过程
    p1 = F.normalize(p1_active, p=2, dim=-1, eps=1e-8)
    z2 = F.normalize(z2_active, p=2, dim=-1, eps=1e-8)
    p2 = F.normalize(p2_active, p=2, dim=-1, eps=1e-8)
    z1 = F.normalize(z1_active, p=2, dim=-1, eps=1e-8)

    # 3. 计算均方误差
    loss1 = ((p1 - z2) ** 2).sum(dim=-1).mean()
    loss2 = ((p2 - z1) ** 2).sum(dim=-1).mean()

    # 4. 【终极防爆】如果出现了 nan 或者 inf，强制清零，防止污染整个梯度
    if torch.isnan(loss1) or torch.isinf(loss1):
        loss1 = torch.tensor(0.0, device=pred_1.device)
    if torch.isnan(loss2) or torch.isinf(loss2):
        loss2 = torch.tensor(0.0, device=pred_2.device)

    # 强制将 loss 截断在合理范围内 (比如最大不超过 10.0)
    loss1 = torch.clamp(loss1, max=10.0)
    loss2 = torch.clamp(loss2, max=10.0)

    return loss1, loss2


def partition_and_correct(logits1, logits2, features, labels, num_classes, attention_mask):
    pred1 = torch.argmax(logits1, dim=-1)
    pred2 = torch.argmax(logits2, dim=-1)
    active_mask = attention_mask == 1

    b_low = (pred1 == labels) & (pred2 == labels) & active_mask
    b_high = ~b_low & active_mask

    prototypes = []
    for c in range(num_classes):
        class_mask = (labels == c) & b_low
        if class_mask.sum() > 0:
            prototypes.append(features[class_mask].mean(dim=0))
        else:
            prototypes.append(torch.zeros(features.size(-1)).to(features.device))
    prototypes = torch.stack(prototypes)

    high_features = features[b_high]
    b_out_mask = torch.zeros_like(b_high)

    if high_features.size(0) > 2:
        dists = torch.cdist(high_features, high_features)
        m = high_features.size(0)
        h_init = high_features.std(dim=0).mean().item() + 1e-6
        h_opt = 1.06 * h_init * (m ** (-0.2))
        density = torch.exp(-(dists ** 2) / (2.0 * (h_opt ** 2))).mean(dim=1)

        n_out = max(1, int(len(density) * Config().ood_noise_ratio))
        _, out_indices = torch.topk(density, n_out, largest=False)

        high_indices = b_high.nonzero(as_tuple=True)
        out_actual_indices = (high_indices[0][out_indices], high_indices[1][out_indices])
        b_out_mask[out_actual_indices] = True
    else:
        b_out_mask = torch.zeros_like(b_high)

    b_high_prime = b_high & ~b_out_mask
    return b_low, b_high_prime, b_out_mask, prototypes


def kl_div(p, q):
    return F.kl_div(p.log(), q, reduction='batchmean')


def kl_div_log(p_log, q):
    return F.kl_div(p_log, q, reduction='batchmean')


def create_mixed_labels(unclean_logits, clean_logits, alpha=0.5):
    min_len = min(unclean_logits.size(0), clean_logits.size(0))
    if min_len == 0:
        return None
    u_logits = unclean_logits[:min_len]
    c_logits = clean_logits[:min_len]
    return alpha * F.softmax(u_logits, dim=1) + (1 - alpha) * F.softmax(c_logits, dim=1)


def seqmix_data(inputs, labels, config):
    alpha = config.mixup_alpha
    lam = np.random.beta(alpha, alpha)

    batch_size = inputs.size(0)
    device = inputs.device
    index = torch.randperm(batch_size).to(device)

    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
    targets_a = labels
    targets_b = labels[index]

    return mixed_inputs, targets_a, targets_b, lam, index


def rbf_kernel(x, gamma_param=None):
    if gamma_param is None:
        gamma_param = 1.0 / x.shape[1]
    x_norm = np.sum(x ** 2, axis=-1).reshape(-1, 1)
    kernel_matrix = np.exp(-gamma_param * (x_norm + x_norm.T - 2.0 * np.dot(x, x.T)))
    return kernel_matrix


def train():
    set_seed(42)
    processor = NerProcessor()
    config = Config()
    processor.clean_output(config)
    writer = SummaryWriter(logdir=os.path.join(config.output_path, "eval"), comment="ner")

    if config.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter.")

    use_gpu = torch.cuda.is_available() and config.use_gpu
    device = torch.device('cuda' if use_gpu else config.device)
    config.device = device
    n_gpu = torch.cuda.device_count()
    logger.info(f"available device: {device}，count_gpu: {n_gpu}")

    # =========================================================================
    # 终极修复：彻底废弃动态扫描，强制硬编码 101 维全量标签，完美对齐 Checkpoint！
    # =========================================================================
    label_list = [
        'B-ACT', 'B-APT', 'B-APT\xa0as', 'B-APT\xa0is', 'B-DOM', 'B-EMAIL', 'B-ENCR',
        'B-FILE', 'B-IDTY', 'B-IDTYL', 'B-IP', 'B-LOC', 'B-MAL', 'B-MD5', 'B-OS',
        'B-PROT', 'B-S-SECTEAM', 'B-SECTEAM', 'B-SHA1', 'B-SHA2', 'B-TIME', 'B-TOOL',
        'B-URL', 'B-VULID', 'B-VULNAME', 'E-ACT', 'E-APT', 'E-APT\xa0as', 'E-APT\xa0is',
        'E-DOM', 'E-EMAIL', 'E-ENCR', 'E-FILE', 'E-IDTY', 'E-IP', 'E-LOC', 'E-MAL',
        'E-MD5', 'E-OS', 'E-PROT', 'E-S-SECTEAM', 'E-SECTEAM', 'E-SHA1', 'E-SHA2',
        'E-TIME', 'E-TOOL', 'E-URL', 'E-VULID', 'E-VULNAME', 'I-ACT', 'I-APT',
        'I-APT\xa0as', 'I-APT\xa0is', 'I-DOM', 'I-EMAIL', 'I-ENCR', 'I-FILE', 'I-IDTY',
        'I-IP', 'I-LOC', 'I-MAL', 'I-MD5', 'I-OS', 'I-PROT', 'I-S-SECTEAM', 'I-SECTEAM',
        'I-SHA1', 'I-SHA2', 'I-TIME', 'I-TOOL', 'I-URL', 'I-VULID', 'I-VULNAME', 'O',
        'PROT', 'S-ACT', 'S-APT', 'S-APT\xa0as', 'S-APT\xa0is', 'S-DOM', 'S-EMAIL',
        'S-ENCR', 'S-FILE', 'S-IDTY', 'S-IP', 'S-LOC', 'S-MAL', 'S-MD5', 'S-OS',
        'S-PROT', 'S-S-SECTEAM', 'S-SECTEAM', 'S-SHA1', 'S-SHA2', 'S-TIME', 'S-TOOL',
        'S-URL', 'S-VULID', 'S-VULNAME', 'also', 'is'
    ]
    label_list.sort()  # 必须排序，保证和训练时生成的顺序严格一致

    print(f"🔥 强行锁定的标签数量为: {len(label_list)}")
    config.label_list = label_list
    num_labels = len(label_list)
    # =========================================================================

    label2id, id2label = processor.get_label2id_id2label(config.output_path, label_list=label_list)

    if config.do_train:
        tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path, do_lower_case=config.do_lower_case)
        bert_config = BertConfig.from_pretrained(config.model_name_or_path, num_labels=num_labels)

        model1 = BERT_BiLSTM_CRF.from_pretrained(config.model_name_or_path, config=bert_config,
                                                 need_birnn=config.need_birnn, rnn_dim=config.rnn_dim)
        model2 = BERT_BiLSTM_CRF.from_pretrained(config.model_name_or_path, config=bert_config,
                                                 need_birnn=config.need_birnn, rnn_dim=config.rnn_dim)
        model1.to(device)
        model2.to(device)

        if use_gpu and n_gpu > 1:
            model1 = torch.nn.DataParallel(model1)
            model2 = torch.nn.DataParallel(model2)

        train_examples, train_features, train_data = processor.get_dataset(config, tokenizer, mode="train")
        train_data_loader = DataLoader(train_data, batch_size=config.train_batch_size,
                                       sampler=RandomSampler(train_data))

        eval_examples, eval_features, eval_data = [], [], None
        if config.do_eval:
            eval_examples, eval_features, eval_data = processor.get_dataset(config, tokenizer, mode="eval")

        optimizer1 = initialize_optimizer(model1, config)
        optimizer2 = initialize_optimizer(model2, config)

        t_total = len(train_data_loader) // config.gradient_accumulation_steps * config.num_train_epochs
        scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=config.warmup_steps,
                                                     num_training_steps=t_total)
        scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=config.warmup_steps,
                                                     num_training_steps=t_total)

        global_step, tr_loss1, tr_loss2, best_f1 = 0, 0.0, 0.0, 0.0

        for ep in range(int(config.num_train_epochs)):
            model1.train()
            model2.train()

            stats_tracker = LabelStatsTracker(id2label)

            pbar = tqdm(train_data_loader, desc=f"🚀 Epoch {ep + 1}/{int(config.num_train_epochs)}", unit="batch")

            for step, batch in enumerate(pbar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, token_type_ids, attention_mask, label_ids = batch

                loss1_crf, logits1, feat1, proj1, pred1 = model1(input_ids, label_ids, token_type_ids, attention_mask)
                loss2_crf, logits2, feat2, proj2, pred2 = model2(input_ids, label_ids, token_type_ids, attention_mask)

                logits1_flat = logits1.view(-1, num_labels)
                logits2_flat = logits2.view(-1, num_labels)
                labels_oh = F.one_hot(label_ids, num_classes=num_labels).float()
                labels_oh_flat = labels_oh.view(-1, num_labels)

                if ep < config.stage1:
                    loss_self1, loss_self2 = self_supervised_loss(pred1, proj2, pred2, proj1, attention_mask == 1)

                    loss1 = loss1_crf.mean() + config.loss_alpha * loss_self1
                    loss2 = loss2_crf.mean() + config.loss_alpha * loss_self2
                else:
                    with torch.no_grad():
                        b_low1, b_high_p1, b_out1, proto1 = partition_and_correct(logits1, logits2, feat1, label_ids,
                                                                                  num_labels, attention_mask)
                        b_low2, b_high_p2, b_out2, proto2 = partition_and_correct(logits2, logits1, feat2, label_ids,
                                                                                  num_labels, attention_mask)

                        total_out_mask = b_out1 | b_out2
                        valid_mask = (attention_mask == 1) & ~total_out_mask

                        stats_tracker.update(label_ids, total_out_mask, attention_mask)

                    b_low1_f, b_low2_f = b_low1.view(-1), b_low2.view(-1)
                    b_high_p1_f, b_high_p2_f = b_high_p1.view(-1), b_high_p2.view(-1)
                    valid_mask_f = valid_mask.view(-1)

                    loss_c1 = cross_entropy(logits1_flat[valid_mask_f],
                                            labels_oh_flat[valid_mask_f]) if valid_mask_f.sum() > 0 else torch.tensor(
                        0.0, device=device)
                    loss_c2 = cross_entropy(logits2_flat[valid_mask_f],
                                            labels_oh_flat[valid_mask_f]) if valid_mask_f.sum() > 0 else torch.tensor(
                        0.0, device=device)

                    mixed_labels1 = create_mixed_labels(logits1_flat[b_high_p1_f], logits1_flat[b_low1_f])
                    mixed_labels2 = create_mixed_labels(logits2_flat[b_high_p2_f], logits2_flat[b_low2_f])

                    loss_u1 = kl_div_log(F.log_softmax(logits1_flat[b_high_p1_f][:mixed_labels1.size(0)], dim=1),
                                         mixed_labels1) if mixed_labels1 is not None else torch.tensor(0.0,
                                                                                                       device=device)
                    loss_u2 = kl_div_log(F.log_softmax(logits2_flat[b_high_p2_f][:mixed_labels2.size(0)], dim=1),
                                         mixed_labels2) if mixed_labels2 is not None else torch.tensor(0.0,
                                                                                                       device=device)

                    loss_cons1, loss_cons2 = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
                    if b_low1_f.sum() > 0:
                        probs2_target = F.softmax(logits2_flat[b_low1_f].detach(), dim=-1)
                        loss_cons1 = kl_div_log(F.log_softmax(logits1_flat[b_low1_f], dim=-1), probs2_target)

                    if b_low2_f.sum() > 0:
                        probs1_target = F.softmax(logits1_flat[b_low2_f].detach(), dim=-1)
                        loss_cons2 = kl_div_log(F.log_softmax(logits2_flat[b_low2_f], dim=-1), probs1_target)

                    loss_self1, loss_self2 = self_supervised_loss(pred1, proj2, pred2, proj1, valid_mask)

                    b_high_p1_seq = b_high_p1.any(dim=1)
                    b_high_p2_seq = b_high_p2.any(dim=1)

                    loss_n1, loss_sim1 = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
                    loss_n2, loss_sim2 = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

                    if b_high_p1_seq.sum() > 0:
                        mixed_inputs1, targets_a1, targets_b1, lam1, idx1 = seqmix_data(input_ids[b_high_p1_seq],
                                                                                        label_ids[b_high_p1_seq],
                                                                                        config)
                        _, logits_mixed1, feat_mixed1, _, _ = model1(mixed_inputs1.long(), targets_a1,
                                                                     token_type_ids[b_high_p1_seq],
                                                                     attention_mask[b_high_p1_seq])

                        mixed_targets1 = lam1 * F.one_hot(targets_a1, num_classes=num_labels).float() + (
                                1 - lam1) * F.one_hot(targets_b1, num_classes=num_labels).float()

                        loss_n1 = kl_div_log(F.log_softmax(logits_mixed1, dim=-1).view(-1, num_labels),
                                             mixed_targets1.view(-1, num_labels))

                        loss_sim1 = semantic_similarity_loss(
                            x_mix=feat_mixed1, x_unclean=feat1[b_high_p1_seq], x_clean=feat1[b_high_p1_seq][idx1],
                            y_mix=mixed_targets1, y_unclean=F.one_hot(targets_a1, num_classes=num_labels).float(),
                            y_clean=F.one_hot(targets_b1, num_classes=num_labels).float()
                        )

                    if b_high_p2_seq.sum() > 0:
                        mixed_inputs2, targets_a2, targets_b2, lam2, idx2 = seqmix_data(input_ids[b_high_p2_seq],
                                                                                        label_ids[b_high_p2_seq],
                                                                                        config)
                        _, logits_mixed2, feat_mixed2, _, _ = model2(mixed_inputs2.long(), targets_a2,
                                                                     token_type_ids[b_high_p2_seq],
                                                                     attention_mask[b_high_p2_seq])

                        mixed_targets2 = lam2 * F.one_hot(targets_a2, num_classes=num_labels).float() + (
                                1 - lam2) * F.one_hot(targets_b2, num_classes=num_labels).float()

                        loss_n2 = kl_div_log(F.log_softmax(logits_mixed2, dim=-1).view(-1, num_labels),
                                             mixed_targets2.view(-1, num_labels))

                        loss_sim2 = semantic_similarity_loss(
                            x_mix=feat_mixed2, x_unclean=feat2[b_high_p2_seq], x_clean=feat2[b_high_p2_seq][idx2],
                            y_mix=mixed_targets2, y_unclean=F.one_hot(targets_a2, num_classes=num_labels).float(),
                            y_clean=F.one_hot(targets_b2, num_classes=num_labels).float()
                        )

                    iter_step = ep - config.stage1 + 1
                    current_gamma = min(1.0, iter_step / 5.0) * config.loss_gamma

                    loss1 = config.loss_beta * loss1_crf.mean() + config.loss_beta * loss_c1 + current_gamma * (
                            loss_u1 + loss_n1 + loss_sim1 + loss_cons1) + config.loss_alpha * loss_self1
                    loss2 = config.loss_beta * loss2_crf.mean() + config.loss_beta * loss_c2 + current_gamma * (
                            loss_u2 + loss_n2 + loss_sim2 + loss_cons2) + config.loss_alpha * loss_self2

                if use_gpu and n_gpu > 1:
                    loss1 = loss1.mean()
                    loss2 = loss2.mean()
                if config.gradient_accumulation_steps > 1:
                    loss1 = loss1 / config.gradient_accumulation_steps
                    loss2 = loss2 / config.gradient_accumulation_steps

                loss1.backward(retain_graph=True)
                loss2.backward()
                tr_loss1 += loss1.item()
                tr_loss2 += loss2.item()

                if (step + 1) % config.gradient_accumulation_steps == 0:
                    # 梯度防爆阀
                    torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)

                    optimizer1.step()
                    optimizer2.step()
                    scheduler1.step()
                    scheduler2.step()
                    model1.zero_grad()
                    model2.zero_grad()
                    global_step += 1

                    pbar.set_postfix({
                        'L1': f"{loss1.item():.4f}",
                        'L2': f"{loss2.item():.4f}",
                        'BestF1': f"{best_f1:.4f}"
                    })

                    if config.logging_steps > 0 and global_step % config.logging_steps == 0:
                        tr_loss_avg1 = tr_loss1 / global_step
                        tr_loss_avg2 = tr_loss2 / global_step
                        writer.add_scalar("Train/loss_model1", tr_loss_avg1, global_step)
                        writer.add_scalar("Train/loss_model2", tr_loss_avg2, global_step)

            if config.do_eval:
                if ep >= config.stage1:
                    stats_tracker.print_stats()

                logger.info(f"📊 第 {ep + 1} 轮训练结束，正在运行评估...")
                overall1, by_type1 = evaluate(config, eval_data, model1, id2label,
                                              [f.ori_tokens for f in eval_features])
                overall2, by_type2 = evaluate(config, eval_data, model2, id2label,
                                              [f.ori_tokens for f in eval_features])

                f1_score1 = overall1.fscore
                f1_score2 = overall2.fscore

                writer.add_scalar("Eval/precision_model1", overall1.prec, ep)
                writer.add_scalar("Eval/precision_model2", overall2.prec, ep)
                writer.add_scalar("Eval/recall_model1", overall1.rec, ep)
                writer.add_scalar("Eval/recall_model2", overall2.rec, ep)
                writer.add_scalar("Eval/f1_score_model1", f1_score1, ep)
                writer.add_scalar("Eval/f1_score_model2", f1_score2, ep)

                logger.info(f"📌 评估结果: 模型1 F1={f1_score1:.4f} | 模型2 F1={f1_score2:.4f}")

                if f1_score1 > best_f1:
                    logger.info(f"******** the best f1 for model1 is {f1_score1}, save model !!! ********")
                    best_f1 = f1_score1
                    save_model(config, model1, tokenizer)

                if f1_score2 > best_f1:
                    logger.info(f"******** the best f1 for model2 is {f1_score2}, save model !!! ********")
                    best_f1 = f1_score2
                    save_model(config, model2, tokenizer)

        writer.close()
        logger.info("NER model training successful!!!")

    if config.do_test:
        tokenizer = BertTokenizer.from_pretrained(config.output_path, do_lower_case=config.do_lower_case)
        # 移除 PyTorch 2.6 安全拦截
        config_dict = torch.load(os.path.join(config.output_path, 'training_config.bin'), weights_only=False)

        # 动态篡改 bert_config 的底层映射字典和维度长度
        bert_config = BertConfig.from_pretrained(config.output_path)
        bert_config.num_labels = num_labels
        bert_config.id2label = id2label
        bert_config.label2id = label2id

        # 传入带有正确维度的 config，强行扩充模型骨架至 101 维
        model1 = BERT_BiLSTM_CRF.from_pretrained(config.output_path,
                                                 config=bert_config,
                                                 need_birnn=config_dict.need_birnn,
                                                 rnn_dim=config_dict.rnn_dim)
        model1.to(device)

        model2 = BERT_BiLSTM_CRF.from_pretrained(config.output_path,
                                                 config=bert_config,
                                                 need_birnn=config_dict.need_birnn,
                                                 rnn_dim=config_dict.rnn_dim)
        model2.to(device)

        test_examples, test_features, test_data = processor.get_dataset(config, tokenizer, mode="test")
        logger.info("====================== Running test ======================")

        all_ori_tokens = [f.ori_tokens for f in test_features]
        all_ori_labels = [e.label.split(" ") for e in test_examples]
        test_sampler = SequentialSampler(test_data)
        test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=config.eval_batch_size)

        model1.eval()
        model2.eval()

        pred_labels = []
        for b_i, (input_ids, token_type_ids, attention_mask, label_ids) in enumerate(
                tqdm(test_data_loader, desc="TestDataLoader", leave=False)):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)

            with torch.no_grad():
                logits1 = model1.predict(input_ids, token_type_ids, attention_mask)
                logits2 = model2.predict(input_ids, token_type_ids, attention_mask)

                for l in logits1:
                    pred_label = []
                    for idx in l:
                        pred_label.append(id2label[idx])
                    pred_labels.append(pred_label)

        # 🔥 修复写测试文件时的空格问题
        with open(os.path.join(config.output_path, "token_labels_test.txt"), "w", encoding="utf-8") as f:
            for ori_tokens, ori_labels, prel in zip(all_ori_tokens, all_ori_labels, pred_labels):
                for ot, ol, pl in zip(ori_tokens, ori_labels, prel):
                    if ot in ["[CLS]", "[SEP]"]:
                        continue
                    else:
                        safe_ot = ot.replace(" ", "_").replace("\xa0", "_")
                        safe_ol = ol.replace(" ", "_").replace("\xa0", "_")
                        safe_pl = pl.replace(" ", "_").replace("\xa0", "_")
                        f.write(f"{safe_ot} {safe_ol} {safe_pl}\n")
                f.write("\n")


def evaluate(config: Config, data, model, id2label, all_ori_tokens):
    ori_labels, pred_labels = [], []
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    sampler = SequentialSampler(data)
    data_loader = DataLoader(data, sampler=sampler, batch_size=config.train_batch_size)
    for b_i, (input_ids, token_type_ids, attention_mask, label_ids) in enumerate(
            tqdm(data_loader, desc="🔍 Evaluating", leave=False)):
        input_ids = input_ids.to(config.device)
        attention_mask = attention_mask.to(config.device)
        token_type_ids = token_type_ids.to(config.device)
        label_ids = label_ids.to(config.device)
        with torch.no_grad():
            logits = model.predict(input_ids, token_type_ids, attention_mask)

        for l in logits:
            pred_labels.append([id2label[idx] for idx in l])

        for l in label_ids:
            ori_labels.append([id2label[idx.item()] for idx in l])

    eval_list = []
    for ori_tokens, oril, prel in zip(all_ori_tokens, ori_labels, pred_labels):
        for ot, ol, pl in zip(ori_tokens, oril, prel):
            if ot in ["[CLS]", "[SEP]"]:
                continue
            # 🔥 核心修复：替换掉词和标签里的“正常空格”与“\xa0”等非法空白符
            safe_ot = ot.replace(" ", "_").replace("\xa0", "_")
            safe_ol = ol.replace(" ", "_").replace("\xa0", "_")
            safe_pl = pl.replace(" ", "_").replace("\xa0", "_")
            eval_list.append(f"{safe_ot} {safe_ol} {safe_pl}\n")
        eval_list.append("\n")

    counts = conlleval.evaluate(eval_list)
    overall, by_type = conlleval.metrics(counts)
    return overall, by_type


def initialize_optimizer(model, config):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
    return optimizer


def save_model(config, model, tokenizer):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(config.output_path)
    tokenizer.save_pretrained(config.output_path)
    torch.save(config, os.path.join(config.output_path, 'training_config.bin'))
    torch.save(model, os.path.join(config.output_path, 'ner_model.ckpt'))
    logger.info("Model and config saved successfully.")


if __name__ == '__main__':
    train()