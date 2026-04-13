#!/usr/bin/env python3
"""Data pipeline and training helpers for BlendEmo."""

import ast
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from labels import LABEL_TO_INDEX

RATIO_FIRST = 0
RATIO_EQUAL = 1
RATIO_SECOND = 2
ALL_MODALITIES = ("hicmae", "wavlm", "openface", "videomae")
MODALITY_DIMS = {
    "hicmae": 2048,
    "wavlm": 1024,
    "openface": 11,
    "videomae": 1408,
}


def parse_enabled_modalities(spec):
    if spec is None:
        return list(ALL_MODALITIES)

    if isinstance(spec, str):
        raw = spec.strip().lower()
        if raw in {"", "all"}:
            return list(ALL_MODALITIES)
        parts = [x.strip() for x in spec.split(",") if x.strip()]
    else:
        parts = [str(x).strip() for x in spec if str(x).strip()]

    normalized = []
    seen = set()
    valid = set(ALL_MODALITIES)
    for name in parts:
        key = str(name).strip().lower()
        if key not in valid:
            raise ValueError(f"Unknown modality: {name}. Valid choices: {sorted(valid)}")
        if key not in seen:
            normalized.append(key)
            seen.add(key)

    if not normalized:
        raise ValueError("At least one modality must be enabled")
    return normalized


def canonical_pair(e1, e2):
    return tuple(sorted([e1, e2]))


def build_set_classes(df):
    singles = sorted(df.loc[df["mix"] == 0, "emotion_1"].dropna().unique().tolist())
    mix_pairs = sorted(
        {
            canonical_pair(r["emotion_1"], r["emotion_2"])
            for _, r in df[df["mix"] == 1].iterrows()
        }
    )

    classes = []
    for emo in singles:
        classes.append(("single", emo))
    for p in mix_pairs:
        classes.append(("mix", p[0], p[1]))

    class_to_idx = {c: i for i, c in enumerate(classes)}
    return classes, class_to_idx


def record_to_structured_targets(record, class_to_idx):
    mix = int(record["mix"])
    e1 = record["emotion_1"]

    if mix == 0:
        set_key = ("single", e1)
        return class_to_idx[set_key], -1

    e2 = record["emotion_2"]
    s1 = float(record["emotion_1_salience"])
    s2 = float(record["emotion_2_salience"])

    pair = canonical_pair(e1, e2)
    set_key = ("mix", pair[0], pair[1])

    if e1 == pair[0]:
        first_sal, second_sal = s1, s2
    else:
        first_sal, second_sal = s2, s1

    if abs(first_sal - second_sal) < 1e-6:
        ratio = RATIO_EQUAL
    elif first_sal > second_sal:
        ratio = RATIO_FIRST
    else:
        ratio = RATIO_SECOND

    return class_to_idx[set_key], ratio


def create_structured_targets(df, class_to_idx):
    set_labels = np.zeros(len(df), dtype=np.int64)
    ratio_labels = np.full(len(df), -1, dtype=np.int64)

    records = df.to_dict(orient="records")
    for i, rec in enumerate(records):
        s, r = record_to_structured_targets(rec, class_to_idx)
        set_labels[i] = s
        ratio_labels[i] = r

    return set_labels, ratio_labels


def _prepare_sequence_array(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr.astype(np.float32)


def _load_openface_array(path):
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        au_str = re.sub(r"\bnan\b", "0", str(row.get("action_units", "[]")))
        try:
            aus = ast.literal_eval(au_str)
            if not isinstance(aus, list):
                aus = [0.0] * 8
        except Exception:
            aus = [0.0] * 8

        feat_row = [
            float(row.get("emotion_index", 0) or 0) / 7.0,
            float(row.get("gaze_yaw", 0) or 0),
            float(row.get("gaze_pitch", 0) or 0),
        ] + list(aus)
        rows.append(feat_row)

    feat = np.array(rows, dtype=np.float32)
    return np.nan_to_num(feat, nan=0.0)


def _load_feature(modality, fname, feature_dirs):
    if modality == "hicmae":
        arr = np.load(os.path.join(feature_dirs["hicmae"], fname + ".npy"))
        return _prepare_sequence_array(arr)

    if modality == "wavlm":
        arr = np.load(os.path.join(feature_dirs["wavlm"], fname + ".npy"))
        return _prepare_sequence_array(arr)

    if modality == "videomae":
        arr = np.load(os.path.join(feature_dirs["videomae"], fname + ".npy"))
        return _prepare_sequence_array(arr)

    if modality == "openface":
        arr = _load_openface_array(os.path.join(feature_dirs["openface3"], fname + ".csv"))
        return _prepare_sequence_array(arr)

    raise KeyError(f"Unknown modality: {modality}")


def _sample_frames(arr, max_frames):
    if max_frames is None or max_frames <= 0 or arr.shape[0] <= max_frames:
        return arr
    idx = np.linspace(0, arr.shape[0] - 1, max_frames, dtype=int)
    return arr[idx]


def compute_feature_stats(
    filenames,
    feature_dirs,
    max_frames_per_file=80,
    enabled_modalities=None,
):
    modalities = parse_enabled_modalities(enabled_modalities)
    stats = {}
    for modality in modalities:
        print(f"    Scanning {modality}...")

        feat_sum = None
        feat_sq_sum = None
        total_count = 0

        for fname in filenames:
            arr = _load_feature(modality, fname, feature_dirs)
            arr = _sample_frames(arr, max_frames_per_file)

            if feat_sum is None:
                feat_sum = arr.sum(axis=0, dtype=np.float64)
                feat_sq_sum = (arr * arr).sum(axis=0, dtype=np.float64)
            else:
                feat_sum += arr.sum(axis=0, dtype=np.float64)
                feat_sq_sum += (arr * arr).sum(axis=0, dtype=np.float64)
            total_count += arr.shape[0]

        if total_count <= 0:
            raise RuntimeError(f"No frames found for modality={modality}")

        mean = feat_sum / float(total_count)
        var = feat_sq_sum / float(total_count) - mean * mean
        var = np.maximum(var, 1e-8)
        std = np.sqrt(var)
        stats[modality] = (mean.astype(np.float32), std.astype(np.float32))

    return stats


class MultiModalDataset(Dataset):
    def __init__(
        self,
        filenames,
        labels,
        mix_labels,
        salience_labels,
        set_labels,
        ratio_labels,
        feature_dirs,
        stats=None,
        enabled_modalities=None,
    ):
        self.filenames = filenames
        self.labels = labels
        self.mix_labels = mix_labels
        self.salience_labels = salience_labels
        self.set_labels = set_labels
        self.ratio_labels = ratio_labels
        self.feature_dirs = feature_dirs
        self.stats = stats
        self.enabled_modalities = set(parse_enabled_modalities(enabled_modalities))

    def __len__(self):
        return len(self.filenames)

    def _normalize(self, feat, key):
        if self.stats and key in self.stats:
            mean, std = self.stats[key]
            feat = (feat - mean) / (std + 1e-8)
        return feat

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        def load_or_dummy(name):
            if name in self.enabled_modalities:
                feat = _load_feature(name, fname, self.feature_dirs)
                return self._normalize(feat, name)
            return np.zeros((1, MODALITY_DIMS[name]), dtype=np.float32)

        hicmae = load_or_dummy("hicmae")
        wavlm = load_or_dummy("wavlm")
        openface = load_or_dummy("openface")
        videomae = load_or_dummy("videomae")

        return {
            "hicmae": torch.from_numpy(hicmae),
            "wavlm": torch.from_numpy(wavlm),
            "openface": torch.from_numpy(openface),
            "videomae": torch.from_numpy(videomae),
            "label": torch.from_numpy(self.labels[idx]),
            "mix_label": torch.tensor(self.mix_labels[idx], dtype=torch.float32),
            "salience_label": torch.tensor(self.salience_labels[idx], dtype=torch.float32),
            "set_label": torch.tensor(self.set_labels[idx], dtype=torch.long),
            "ratio_label": torch.tensor(self.ratio_labels[idx], dtype=torch.long),
            "filename": fname,
        }


def pad_sequence(tensors):
    max_len = max(t.shape[0] for t in tensors)
    dim = tensors[0].shape[1]

    padded = torch.zeros(len(tensors), max_len, dim, dtype=tensors[0].dtype)
    masks = torch.zeros(len(tensors), max_len, dtype=torch.bool)

    for i, t in enumerate(tensors):
        seq_len = t.shape[0]
        padded[i, :seq_len] = t
        masks[i, :seq_len] = True

    return padded, masks


def collate_batch(batch):
    hicmae, hicmae_mask = pad_sequence([b["hicmae"] for b in batch])
    wavlm, wavlm_mask = pad_sequence([b["wavlm"] for b in batch])
    openface, openface_mask = pad_sequence([b["openface"] for b in batch])
    videomae, videomae_mask = pad_sequence([b["videomae"] for b in batch])

    return {
        "hicmae": hicmae,
        "wavlm": wavlm,
        "openface": openface,
        "videomae": videomae,
        "hicmae_mask": hicmae_mask,
        "wavlm_mask": wavlm_mask,
        "openface_mask": openface_mask,
        "videomae_mask": videomae_mask,
        "label": torch.stack([b["label"] for b in batch]),
        "mix_label": torch.stack([b["mix_label"] for b in batch]),
        "salience_label": torch.stack([b["salience_label"] for b in batch]),
        "set_label": torch.stack([b["set_label"] for b in batch]),
        "ratio_label": torch.stack([b["ratio_label"] for b in batch]),
        "filename": [b["filename"] for b in batch],
    }


def _forward_model(model, batch, device):
    return model(
        hicmae=batch["hicmae"].to(device),
        wavlm=batch["wavlm"].to(device),
        openface=batch["openface"].to(device),
        videomae=batch["videomae"].to(device),
        hicmae_mask=batch["hicmae_mask"].to(device),
        wavlm_mask=batch["wavlm_mask"].to(device),
        openface_mask=batch["openface_mask"].to(device),
        videomae_mask=batch["videomae_mask"].to(device),
    )


def info_nce_loss(anchor, target, temperature=0.2):
    # anchor/target: (B, D)
    if anchor.size(0) <= 1:
        return torch.tensor(0.0, device=anchor.device)

    a = F.normalize(anchor, dim=-1)
    t = F.normalize(target, dim=-1)
    logits = torch.matmul(a, t.t()) / max(1e-6, float(temperature))
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return F.cross_entropy(logits, labels)


def shared_private_loss(shared_tokens, private_tokens, token_mask=None):
    # shared/private: (B, M, D)
    num_mod = int(shared_tokens.size(1))
    device = shared_tokens.device
    if token_mask is None:
        token_mask = torch.ones(
            shared_tokens.size(0),
            num_mod,
            device=device,
            dtype=torch.bool,
        )

    align_terms = []
    for i in range(num_mod):
        for j in range(i + 1, num_mod):
            pair_mask = token_mask[:, i] & token_mask[:, j]
            if pair_mask.any():
                align_terms.append(F.mse_loss(shared_tokens[pair_mask, i], shared_tokens[pair_mask, j]))

    if align_terms:
        align = torch.stack(align_terms).mean()
    else:
        align = torch.tensor(0.0, device=device)

    flat_mask = token_mask.reshape(-1)
    if flat_mask.any():
        cos_abs = F.cosine_similarity(
            shared_tokens.reshape(-1, shared_tokens.size(-1))[flat_mask],
            private_tokens.reshape(-1, private_tokens.size(-1))[flat_mask],
            dim=-1,
        ).abs().mean()
    else:
        cos_abs = torch.tensor(0.0, device=device)

    return align + cos_abs, align, cos_abs
def _pick_indices(mask, limit, seed):
    idx = np.where(mask)[0]
    if limit is not None and limit > 0 and len(idx) > limit:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=limit, replace=False))
    return idx
