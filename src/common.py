"""Common training utilities for BlendEmo."""

import random

import numpy as np
import torch
import torch.nn.functional as F

from labels import LABEL_TO_INDEX


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_labels(records):
    labels = np.zeros((len(records), 6), dtype=np.float32)
    mix_labels = np.zeros(len(records), dtype=np.float32)
    salience_labels = np.zeros(len(records), dtype=np.float32)

    for i, record in enumerate(records):
        e1 = record["emotion_1"]
        e2 = record["emotion_2"]
        s1 = record["emotion_1_salience"]
        s2 = record["emotion_2_salience"]
        mix = record["mix"]
        if mix == 0:
            labels[i, LABEL_TO_INDEX[e1]] = 1.0
            mix_labels[i] = 0.0
            salience_labels[i] = 1.0
        else:
            labels[i, LABEL_TO_INDEX[e1]] = s1 / 100.0
            labels[i, LABEL_TO_INDEX[e2]] = s2 / 100.0
            mix_labels[i] = 1.0
            salience_labels[i] = max(s1, s2) / 100.0

    return labels, mix_labels, salience_labels


def mixup_data(batch, alpha=0.2):
    if alpha <= 0:
        return batch
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)

    batch_size = batch["label"].shape[0]
    index = torch.randperm(batch_size)

    mixed_batch = {}
    for key in batch:
        if key == "filename":
            mixed_batch[key] = batch[key]
        elif key.endswith("_mask"):
            mixed_batch[key] = batch[key] | batch[key][index]
        else:
            mixed_batch[key] = lam * batch[key] + (1 - lam) * batch[key][index]
    return mixed_batch


def label_smoothing(labels, epsilon=0.05):
    k = labels.shape[1]
    return (1 - epsilon) * labels + epsilon / k


def focal_kl_loss(log_probs, targets, gamma=2.0):
    kl_per_sample = F.kl_div(log_probs, targets, reduction="none").sum(dim=1)
    with torch.no_grad():
        weight = (1 - torch.exp(-kl_per_sample)) ** gamma
    return (weight * kl_per_sample).mean()


def rdrop_kl(logits1, logits2):
    p = F.log_softmax(logits1, dim=1)
    q = F.log_softmax(logits2, dim=1)
    p_soft = F.softmax(logits1, dim=1)
    q_soft = F.softmax(logits2, dim=1)
    kl1 = F.kl_div(p, q_soft, reduction="batchmean")
    kl2 = F.kl_div(q, p_soft, reduction="batchmean")
    return (kl1 + kl2) / 2.0
