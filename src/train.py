#!/usr/bin/env python3
"""BlendEmo trainer."""

import argparse
import math
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import create_labels, mixup_data, set_seed
from common import focal_kl_loss, label_smoothing, rdrop_kl
from data import (
    RATIO_EQUAL,
    RATIO_FIRST,
    MultiModalDataset,
    _forward_model,
    _pick_indices,
    build_set_classes,
    collate_batch,
    compute_feature_stats,
    create_structured_targets,
    info_nce_loss,
    parse_enabled_modalities,
    shared_private_loss,
)
from model import (
    BlendEmoModel,
)
from metrics import acc_presence_total, acc_salience_total


RATIO_SECOND = 2


def mixup_data_blendemo(batch, alpha=0.2):
    if alpha <= 0:
        return batch

    keep_keys = {"set_label", "ratio_label", "filename"}
    raw = {k: v for k, v in batch.items() if k in keep_keys}
    mutable = {k: v for k, v in batch.items() if k not in keep_keys}
    mixed = mixup_data(mutable, alpha=alpha)
    mixed.update(raw)
    return mixed


def masked_pair_ratio_ce(ratio_logits, pair_idx, ratio_target):
    mask = ratio_target != -1
    if not mask.any():
        return torch.tensor(0.0, device=ratio_logits.device)
    selected = ratio_logits[mask, pair_idx[mask], :]
    return F.cross_entropy(selected, ratio_target[mask])


def masked_pair_ratio_rdrop_kl(ratio_logits1, ratio_logits2, pair_idx, ratio_mask):
    if not ratio_mask.any():
        return torch.tensor(0.0, device=ratio_logits1.device)
    idx = pair_idx[ratio_mask]
    l1 = ratio_logits1[ratio_mask, idx, :]
    l2 = ratio_logits2[ratio_mask, idx, :]
    return rdrop_kl(l1, l2)


def decode_blendemo_predictions(
    set_logits,
    ratio_logits,
    filenames,
    set_classes,
    num_single_classes,
):
    set_pred = np.argmax(set_logits, axis=1)
    preds = {}

    for i, fname in enumerate(filenames):
        cls_idx = int(set_pred[i])
        state = set_classes[cls_idx]

        if cls_idx < num_single_classes or state[0] == "single":
            preds[fname] = [{"emotion": state[1], "salience": 100.0}]
            continue

        pair_idx = cls_idx - num_single_classes
        ratio_pred = int(np.argmax(ratio_logits[i, pair_idx]))
        e1, e2 = state[1], state[2]
        if ratio_pred == RATIO_EQUAL:
            s1, s2 = 50.0, 50.0
        elif ratio_pred == RATIO_FIRST:
            s1, s2 = 70.0, 30.0
        else:
            s1, s2 = 30.0, 70.0

        preds[fname] = [
            {"emotion": e1, "salience": s1},
            {"emotion": e2, "salience": s2},
        ]

    return preds


def evaluate_blendemo(model, loader, device, set_classes, num_single_classes):
    model.eval()

    all_filenames = []
    all_set_logits = []
    all_ratio_logits = []

    with torch.no_grad():
        for batch in loader:
            out = _forward_model(model, batch, device)
            all_set_logits.append(out["set_logits"].cpu().numpy())
            all_ratio_logits.append(out["ratio_logits"].cpu().numpy())
            all_filenames.extend(batch["filename"])

    all_set_logits = np.concatenate(all_set_logits, axis=0)
    all_ratio_logits = np.concatenate(all_ratio_logits, axis=0)

    preds = decode_blendemo_predictions(
        all_set_logits,
        all_ratio_logits,
        all_filenames,
        set_classes,
        num_single_classes,
    )
    acc_p = acc_presence_total(preds)
    acc_s = acc_salience_total(preds)
    return {
        "acc_presence": acc_p,
        "acc_salience": acc_s,
        "score": (acc_p + acc_s) / 2.0,
        "method": "blendemo",
    }


def train_one_epoch_blendemo(
    model,
    loader,
    optimizer,
    device,
    num_single_classes,
    use_mixup=True,
    mixup_alpha=0.2,
    set_loss_weight=0.45,
    ratio_loss_weight=0.25,
    soft_loss_weight=0.9,
    mix_loss_weight=0.4,
    salience_loss_weight=0.35,
    rdrop_weight=0.8,
    sp_loss_weight=0.0,
    mi_loss_weight=0.0,
    graph_sparse_weight=0.0,
    router_balance_weight=0.0,
    audio_gate_reg_weight=0.0,
    mi_temperature=0.2,
    label_smooth_eps=0.05,
    focal_gamma=2.0,
):
    model.train()

    bce_criterion = nn.BCEWithLogitsLoss()
    mse_criterion = nn.MSELoss()

    stats = {
        "loss": 0.0,
        "set": 0.0,
        "ratio": 0.0,
        "soft": 0.0,
        "mix": 0.0,
        "sal": 0.0,
        "rdrop": 0.0,
        "sp": 0.0,
        "align": 0.0,
        "ortho": 0.0,
        "mi": 0.0,
        "graph_sparse": 0.0,
        "router_bal": 0.0,
        "audio_gate": 0.0,
    }

    for raw_batch in tqdm(loader, desc="Training", leave=False):
        batch = mixup_data_blendemo(raw_batch, alpha=mixup_alpha) if use_mixup else raw_batch

        label = batch["label"].to(device)
        mix_label = batch["mix_label"].to(device)
        salience_label = batch["salience_label"].to(device)
        set_label = batch["set_label"].to(device)
        ratio_label = batch["ratio_label"].to(device)
        ratio_mask = ratio_label != -1
        pair_index = (set_label - int(num_single_classes)).clamp_min(0)

        optimizer.zero_grad(set_to_none=True)

        out1 = _forward_model(model, batch, device)
        out2 = _forward_model(model, batch, device)
        token_mask = out1["token_mask"]

        set_loss = (
            F.cross_entropy(out1["set_logits"], set_label)
            + F.cross_entropy(out2["set_logits"], set_label)
        ) / 2.0

        ratio_loss = (
            masked_pair_ratio_ce(out1["ratio_logits"], pair_index, ratio_label)
            + masked_pair_ratio_ce(out2["ratio_logits"], pair_index, ratio_label)
        ) / 2.0

        smooth_label = label_smoothing(label, epsilon=label_smooth_eps)
        soft_loss = (
            focal_kl_loss(torch.log_softmax(out1["emotion_logits"], dim=1), smooth_label, focal_gamma)
            + focal_kl_loss(torch.log_softmax(out2["emotion_logits"], dim=1), smooth_label, focal_gamma)
        ) / 2.0

        mix_loss = (
            bce_criterion(out1["mix_logit"].squeeze(-1), mix_label)
            + bce_criterion(out2["mix_logit"].squeeze(-1), mix_label)
        ) / 2.0

        sal_pred1 = torch.sigmoid(out1["salience_logit"].squeeze(-1))
        sal_pred2 = torch.sigmoid(out2["salience_logit"].squeeze(-1))
        sal_loss = (
            mse_criterion(sal_pred1, salience_label)
            + mse_criterion(sal_pred2, salience_label)
        ) / 2.0

        rdrop_loss = rdrop_kl(out1["emotion_logits"], out2["emotion_logits"])
        rdrop_loss = rdrop_loss + 0.5 * rdrop_kl(out1["set_logits"], out2["set_logits"])
        rdrop_loss = rdrop_loss + 0.25 * masked_pair_ratio_rdrop_kl(
            out1["ratio_logits"],
            out2["ratio_logits"],
            pair_index,
            ratio_mask,
        )

        sp1, align1, ortho1 = shared_private_loss(
            out1["shared_tokens"],
            out1["private_tokens"],
            token_mask=out1["token_mask"],
        )
        sp2, align2, ortho2 = shared_private_loss(
            out2["shared_tokens"],
            out2["private_tokens"],
            token_mask=out2["token_mask"],
        )
        sp_loss = (sp1 + sp2) / 2.0
        align_loss = (align1 + align2) / 2.0
        ortho_loss = (ortho1 + ortho2) / 2.0

        p1 = out1["private_tokens"]
        p2 = out2["private_tokens"]
        valid_modalities = token_mask.any(dim=0)
        num_mod = int(valid_modalities.sum().item())

        mi1 = torch.tensor(0.0, device=device)
        mi2 = torch.tensor(0.0, device=device)
        for m_idx in range(p1.size(1)):
            valid_rows = token_mask[:, m_idx]
            if not valid_rows.any():
                continue
            mi1 = mi1 + info_nce_loss(
                out1["fused_feature"][valid_rows],
                p1[valid_rows, m_idx],
                temperature=mi_temperature,
            )
            mi2 = mi2 + info_nce_loss(
                out2["fused_feature"][valid_rows],
                p2[valid_rows, m_idx],
                temperature=mi_temperature,
            )
        if num_mod > 0:
            mi1 = mi1 / float(num_mod)
            mi2 = mi2 / float(num_mod)
        mi_loss = (mi1 + mi2) / 2.0

        rel = (out1["relation_weights"] + out2["relation_weights"]) / 2.0
        rel_entropy = -(rel * torch.log(rel.clamp_min(1e-8))).sum(dim=-1)
        rel_norm = math.log(float(max(2, rel.size(-1) - 1)))
        graph_sparse_rows = rel_entropy / rel_norm
        if token_mask.any():
            graph_sparse = graph_sparse_rows[token_mask].mean()
        else:
            graph_sparse = torch.tensor(0.0, device=device)

        router_weights = (out1["router_weights"] + out2["router_weights"]) / 2.0
        target = token_mask.to(dtype=router_weights.dtype)
        target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
        router_balance = (router_weights.mean(dim=0) - target.mean(dim=0)).pow(2).mean()

        audio_gates = (out1["audio_gates"] + out2["audio_gates"]) / 2.0
        if token_mask.any():
            audio_gate_reg = (audio_gates[token_mask] - 0.5).abs().mean()
        else:
            audio_gate_reg = torch.tensor(0.0, device=device)

        loss = (
            set_loss_weight * set_loss
            + ratio_loss_weight * ratio_loss
            + soft_loss_weight * soft_loss
            + mix_loss_weight * mix_loss
            + salience_loss_weight * sal_loss
            + rdrop_weight * rdrop_loss
            + sp_loss_weight * sp_loss
            + mi_loss_weight * mi_loss
            + graph_sparse_weight * graph_sparse
            + router_balance_weight * router_balance
            + audio_gate_reg_weight * audio_gate_reg
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        stats["loss"] += float(loss.item())
        stats["set"] += float(set_loss.item())
        stats["ratio"] += float(ratio_loss.item())
        stats["soft"] += float(soft_loss.item())
        stats["mix"] += float(mix_loss.item())
        stats["sal"] += float(sal_loss.item())
        stats["rdrop"] += float(rdrop_loss.item())
        stats["sp"] += float(sp_loss.item())
        stats["align"] += float(align_loss.item())
        stats["ortho"] += float(ortho_loss.item())
        stats["mi"] += float(mi_loss.item())
        stats["graph_sparse"] += float(graph_sparse.item())
        stats["router_bal"] += float(router_balance.item())
        stats["audio_gate"] += float(audio_gate_reg.item())

    n = len(loader)
    return {k: v / n for k, v in stats.items()}


def run_train_blendemo(config):
    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    print(f"Device: {device}")
    print("=== BlendEmo Config ===")
    for k, v in config.items():
        print(f"  {k}: {v}")

    data_dir = config["data_dir"]
    feature_base = os.path.join(data_dir, "pre_extracted_train_data")
    feature_dirs = {
        "hicmae": os.path.join(feature_base, "HiCMAE_train"),
        "wavlm": os.path.join(feature_base, "WavLM_large_train"),
        "openface3": os.path.join(feature_base, "OpenFace_3_train"),
        "videomae": os.path.join(feature_base, "VideoMAEv2_train"),
    }

    df = pd.read_csv(os.path.join(data_dir, "train_metadata.csv"))
    labels, mix_labels, salience_labels = create_labels(df.to_dict(orient="records"))
    set_classes, class_to_idx = build_set_classes(df)
    set_all, ratio_all = create_structured_targets(df, class_to_idx)
    enabled_modalities = parse_enabled_modalities(config.get("enabled_modalities"))
    num_single_classes = sum(1 for cls in set_classes if cls[0] == "single")

    print(f"Samples: {len(df)}")
    print(f"Set classes: {len(set_classes)}")
    print(f"Single classes: {num_single_classes}")
    print(f"Pair classes: {len(set_classes) - num_single_classes}")
    print(f"Enabled modalities: {enabled_modalities}")

    folds = config.get("folds", sorted(df["fold"].unique().tolist()))
    all_results = []

    for fold_id in folds:
        print(f"\n{'=' * 60}\nFold {fold_id}\n{'=' * 60}")

        train_mask = (df["fold"] != fold_id).to_numpy()
        val_mask = ~train_mask
        train_idx = _pick_indices(train_mask, config.get("max_train_samples"), 100 + int(fold_id))
        val_idx = _pick_indices(val_mask, config.get("max_val_samples"), 200 + int(fold_id))

        train_files = df.iloc[train_idx]["filename"].tolist()
        val_files = df.iloc[val_idx]["filename"].tolist()

        print(f"Train: {len(train_files)}, Val: {len(val_files)}")
        print("  Computing feature statistics...")
        stats = compute_feature_stats(
            train_files,
            feature_dirs,
            max_frames_per_file=config.get("stats_max_frames_per_file", 80),
            enabled_modalities=enabled_modalities,
        )

        train_dataset = MultiModalDataset(
            train_files,
            labels[train_idx],
            mix_labels[train_idx],
            salience_labels[train_idx],
            set_all[train_idx],
            ratio_all[train_idx],
            feature_dirs,
            stats=stats,
            enabled_modalities=enabled_modalities,
        )
        val_dataset = MultiModalDataset(
            val_files,
            labels[val_idx],
            mix_labels[val_idx],
            salience_labels[val_idx],
            set_all[val_idx],
            ratio_all[val_idx],
            feature_dirs,
            stats=stats,
            enabled_modalities=enabled_modalities,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.get("batch_size", 24),
            shuffle=True,
            num_workers=config.get("num_workers", 4),
            collate_fn=collate_batch,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get("batch_size", 24),
            shuffle=False,
            num_workers=config.get("num_workers", 4),
            collate_fn=collate_batch,
        )

        model = BlendEmoModel(
            set_classes=set_classes,
            hicmae_dim=2048,
            wavlm_dim=1024,
            openface_dim=11,
            videomae_dim=1408,
            hidden_dim=config.get("hidden_dim", 192),
            dropout=config.get("dropout", 0.35),
            lmf_rank=config.get("lmf_rank", 10),
            prompt_layers=config.get("prompt_layers", 1),
            enabled_modalities=enabled_modalities,
            ratio_mode=config.get("ratio_mode", "pair_conditioned"),
            ratio_context_mode=config.get("ratio_context_mode", "full"),
            disable_pair_embedding_branch=config.get("disable_pair_embedding_branch", False),
            disable_pair_interaction=config.get("disable_pair_interaction", False),
            disable_emotion_prior=config.get("disable_emotion_prior", False),
            disable_setmix_prior=config.get("disable_setmix_prior", False),
            disable_audio_mixer=config.get("disable_audio_mixer", False),
            disable_prompt_fusion=config.get("disable_prompt_fusion", False),
            disable_relation_fusion=config.get("disable_relation_fusion", False),
            router_pool_mode=config.get("router_pool_mode", "router"),
            disable_lmf_branch=config.get("disable_lmf_branch", False),
        ).to(device)

        if fold_id == folds[0]:
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("lr", 3e-4),
            weight_decay=config.get("weight_decay", 6e-3),
        )

        num_epochs = config.get("num_epochs", 40)
        warmup = config.get("warmup_epochs", 3)

        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / max(1, warmup)
            progress = (epoch - warmup) / max(1, num_epochs - warmup)
            return max(0.01, 0.5 * (1 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        selection_mode = config.get("selection_mode", "best")
        fixed_epoch = config.get("fixed_epoch", None)
        if selection_mode == "fixed_epoch":
            if fixed_epoch is None:
                raise ValueError("fixed_epoch must be set when selection_mode='fixed_epoch'")
            fixed_epoch = int(fixed_epoch)
            if fixed_epoch < 1 or fixed_epoch > num_epochs:
                raise ValueError(f"fixed_epoch must be in [1, {num_epochs}], got {fixed_epoch}")
        elif selection_mode != "best":
            raise ValueError(f"Unsupported selection_mode: {selection_mode}")

        best_score = -1.0
        best_results = None
        patience_counter = 0
        patience = config.get("patience", 12)

        for epoch in range(num_epochs):
            lr_now = optimizer.param_groups[0]["lr"]
            tr = train_one_epoch_blendemo(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                num_single_classes=num_single_classes,
                use_mixup=config.get("use_mixup", True),
                mixup_alpha=config.get("mixup_alpha", 0.2),
                set_loss_weight=config.get("set_loss_weight", 0.45),
                ratio_loss_weight=config.get("ratio_loss_weight", 0.25),
                soft_loss_weight=config.get("soft_loss_weight", 0.9),
                mix_loss_weight=config.get("mix_loss_weight", 0.4),
                salience_loss_weight=config.get("salience_loss_weight", 0.35),
                rdrop_weight=config.get("rdrop_weight", 0.8),
                sp_loss_weight=config.get("sp_loss_weight", 0.0),
                mi_loss_weight=config.get("mi_loss_weight", 0.0),
                graph_sparse_weight=config.get("graph_sparse_weight", 0.0),
                router_balance_weight=config.get("router_balance_weight", 0.0),
                audio_gate_reg_weight=config.get("audio_gate_reg_weight", 0.0),
                mi_temperature=config.get("mi_temperature", 0.2),
                label_smooth_eps=config.get("label_smooth_eps", 0.05),
                focal_gamma=config.get("focal_gamma", 2.0),
            )
            scheduler.step()

            val_stats = evaluate_blendemo(
                model=model,
                loader=val_loader,
                device=device,
                set_classes=set_classes,
                num_single_classes=num_single_classes,
            )

            marker = ""
            current_epoch = epoch + 1
            if selection_mode == "fixed_epoch":
                if current_epoch == fixed_epoch:
                    best_score = val_stats["score"]
                    best_results = val_stats.copy()
                    best_results["epoch"] = current_epoch
                    torch.save(
                        model.state_dict(),
                        os.path.join(output_dir, "checkpoints", f"best_fold{fold_id}.pt"),
                    )
                    marker = " ★fixed"
            else:
                if val_stats["score"] > best_score:
                    best_score = val_stats["score"]
                    best_results = val_stats.copy()
                    best_results["epoch"] = current_epoch
                    patience_counter = 0
                    torch.save(
                        model.state_dict(),
                        os.path.join(output_dir, "checkpoints", f"best_fold{fold_id}.pt"),
                    )
                    marker = " ★"
                else:
                    patience_counter += 1

            print(
                f"E{current_epoch:2d} lr={lr_now:.5f} "
                f"L={tr['loss']:.4f}(set={tr['set']:.4f}+ratio={tr['ratio']:.4f}+"
                f"soft={tr['soft']:.4f}+mix={tr['mix']:.4f}+sal={tr['sal']:.4f}+"
                f"rd={tr['rdrop']:.4f}+sp={tr['sp']:.4f}+mi={tr['mi']:.4f}+"
                f"gs={tr['graph_sparse']:.4f}+rb={tr['router_bal']:.4f}+ag={tr['audio_gate']:.4f}) | "
                f"P={val_stats['acc_presence']:.4f} S={val_stats['acc_salience']:.4f} "
                f"Score={val_stats['score']:.4f}({val_stats['method']}){marker}"
            )

            if selection_mode == "best" and patience_counter >= patience:
                print(f"Early stopping at epoch {current_epoch}")
                break

        assert best_results is not None
        all_results.append(
            {
                "fold": int(fold_id),
                "acc_presence": best_results["acc_presence"],
                "acc_salience": best_results["acc_salience"],
                "score": best_results["score"],
                "best_epoch": int(best_results["epoch"]),
                "method": best_results["method"],
            }
        )

        summary_tag = "fixed" if selection_mode == "fixed_epoch" else "best"
        print(
            f"Fold {fold_id} {summary_tag}: P={best_results['acc_presence']:.4f} "
            f"S={best_results['acc_salience']:.4f} Score={best_results['score']:.4f} "
            f"(E{best_results['epoch']}, {best_results['method']})"
        )

    results_df = pd.DataFrame(all_results)
    print("\n" + "=" * 60)
    print("FINAL RESULTS: BLENDEMO")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(
        f"\nACC Presence: {results_df['acc_presence'].mean():.4f} ± {results_df['acc_presence'].std():.4f}\n"
        f"ACC Salience: {results_df['acc_salience'].mean():.4f} ± {results_df['acc_salience'].std():.4f}\n"
        f"Score:        {results_df['score'].mean():.4f} ± {results_df['score'].std():.4f}"
    )

    out_csv = os.path.join(output_dir, "results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")
    return results_df


def build_config(args):
    config = {
        "seed": 42,
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "folds": [0, 1, 2, 3, 4],
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "lmf_rank": args.lmf_rank,
        "prompt_layers": args.prompt_layers,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "stats_max_frames_per_file": args.stats_max_frames_per_file,
        "selection_mode": args.selection_mode,
        "fixed_epoch": args.fixed_epoch,
        "enabled_modalities": parse_enabled_modalities(args.enabled_modalities),
        "ratio_mode": args.ratio_mode,
        "ratio_context_mode": args.ratio_context_mode,
        "disable_pair_embedding_branch": args.disable_pair_embedding_branch,
        "disable_pair_interaction": args.disable_pair_interaction,
        "disable_emotion_prior": args.disable_emotion_prior,
        "disable_setmix_prior": args.disable_setmix_prior,
        "disable_audio_mixer": args.disable_audio_mixer,
        "disable_prompt_fusion": args.disable_prompt_fusion,
        "disable_relation_fusion": args.disable_relation_fusion,
        "router_pool_mode": args.router_pool_mode,
        "disable_lmf_branch": args.disable_lmf_branch,
        "use_mixup": not args.disable_mixup,
        "mixup_alpha": args.mixup_alpha,
        "set_loss_weight": args.set_loss_weight,
        "ratio_loss_weight": args.ratio_loss_weight,
        "soft_loss_weight": args.soft_loss_weight,
        "mix_loss_weight": args.mix_loss_weight,
        "salience_loss_weight": args.salience_loss_weight,
        "rdrop_weight": args.rdrop_weight,
        "sp_loss_weight": args.sp_loss_weight,
        "mi_loss_weight": args.mi_loss_weight,
        "graph_sparse_weight": args.graph_sparse_weight,
        "router_balance_weight": args.router_balance_weight,
        "audio_gate_reg_weight": args.audio_gate_reg_weight,
        "mi_temperature": args.mi_temperature,
        "label_smooth_eps": args.label_smooth_eps,
        "focal_gamma": args.focal_gamma,
    }

    if args.folds is not None:
        config["folds"] = [int(x) for x in args.folds.split(",") if x.strip()]

    if args.smoke:
        config.update(
            {
                "folds": [0],
                "batch_size": 8,
                "num_workers": 0,
                "num_epochs": 2,
                "warmup_epochs": 1,
                "patience": 2,
                "max_train_samples": 160,
                "max_val_samples": 96,
                "stats_max_frames_per_file": 40,
            }
        )

    return config


def main():
    RELEASE_ROOT = os.path.dirname(SRC_DIR)
    BLEMORE_ROOT = os.path.dirname(RELEASE_ROOT)
    parser = argparse.ArgumentParser(description="BlendEmo trainer")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(BLEMORE_ROOT, "ble-datasets"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(RELEASE_ROOT, "outputs_cv"),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--folds", type=str, default=None)

    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--lmf_rank", type=int, default=10)
    parser.add_argument("--prompt_layers", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=6e-3)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--stats_max_frames_per_file", type=int, default=80)
    parser.add_argument(
        "--enabled_modalities",
        type=str,
        default="all",
        help="Comma-separated subset from {hicmae,wavlm,openface,videomae}. Use 'all' for full model.",
    )
    parser.add_argument(
        "--ratio_mode",
        type=str,
        default="pair_conditioned",
        choices=["pair_conditioned", "global_v44"],
        help="Ratio branch type: BlendEmo pair-conditioned or legacy global ratio head.",
    )
    parser.add_argument(
        "--ratio_context_mode",
        type=str,
        default="full",
        choices=["full", "z_add_only"],
        help="Context source for pair-conditioned ratio head.",
    )
    parser.add_argument(
        "--disable_pair_embedding_branch",
        action="store_true",
        help="Remove pair token branch from the BlendEmo ratio head.",
    )
    parser.add_argument(
        "--disable_pair_interaction",
        action="store_true",
        help="Keep pair token features but remove context-pair multiplicative interaction.",
    )
    parser.add_argument(
        "--disable_emotion_prior",
        action="store_true",
        help="Zero out emotion-derived scalar priors in the BlendEmo ratio head.",
    )
    parser.add_argument(
        "--disable_setmix_prior",
        action="store_true",
        help="Zero out set/mix-derived scalar priors in the BlendEmo ratio head.",
    )
    parser.add_argument(
        "--disable_audio_mixer",
        action="store_true",
        help="Bypass the V44 audio-guided residual mixer in the shared-token trunk.",
    )
    parser.add_argument(
        "--disable_prompt_fusion",
        action="store_true",
        help="Bypass prompt-token transformer fusion and forward mixed tokens directly.",
    )
    parser.add_argument(
        "--disable_relation_fusion",
        action="store_true",
        help="Bypass relation-graph fusion and keep prompt tokens unchanged.",
    )
    parser.add_argument(
        "--router_pool_mode",
        type=str,
        default="router",
        choices=["router", "mean"],
        help="Pooling mode after the relation trunk: learned router or masked mean.",
    )
    parser.add_argument(
        "--disable_lmf_branch",
        action="store_true",
        help="Zero out the multiplicative private branch z_mul and its rank features.",
    )
    parser.add_argument(
        "--selection_mode",
        type=str,
        default="best",
        choices=["best", "fixed_epoch"],
    )
    parser.add_argument("--fixed_epoch", type=int, default=None)

    parser.add_argument("--disable_mixup", action="store_true")
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--set_loss_weight", type=float, default=0.45)
    parser.add_argument("--ratio_loss_weight", type=float, default=0.25)
    parser.add_argument("--soft_loss_weight", type=float, default=0.9)
    parser.add_argument("--mix_loss_weight", type=float, default=0.4)
    parser.add_argument("--salience_loss_weight", type=float, default=0.35)
    parser.add_argument("--rdrop_weight", type=float, default=0.8)
    parser.add_argument("--sp_loss_weight", type=float, default=0.0)
    parser.add_argument("--mi_loss_weight", type=float, default=0.0)
    parser.add_argument("--graph_sparse_weight", type=float, default=0.0)
    parser.add_argument("--router_balance_weight", type=float, default=0.0)
    parser.add_argument("--audio_gate_reg_weight", type=float, default=0.0)
    parser.add_argument("--mi_temperature", type=float, default=0.2)
    parser.add_argument("--label_smooth_eps", type=float, default=0.05)
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    args = parser.parse_args()
    config = build_config(args)
    run_train_blendemo(config)


if __name__ == "__main__":
    main()
