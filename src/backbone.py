#!/usr/bin/env python3
"""Core four-modality backbone for BlendEmo."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import LightEncoder, SeqEncoder


def _normalize_enabled_modalities(enabled_modalities, all_names):
    if enabled_modalities is None:
        return tuple(all_names)

    if isinstance(enabled_modalities, str):
        enabled_modalities = [x.strip() for x in enabled_modalities.split(",") if x.strip()]

    normalized = []
    seen = set()
    valid = set(all_names)
    for name in enabled_modalities:
        key = str(name).strip().lower()
        if key not in valid:
            raise ValueError(f"Unknown modality: {name}. Valid choices: {sorted(valid)}")
        if key not in seen:
            normalized.append(key)
            seen.add(key)

    if not normalized:
        raise ValueError("At least one modality must be enabled")
    return tuple(normalized)


def _masked_mean(tokens, token_mask):
    weights = token_mask.unsqueeze(-1).to(tokens.dtype)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denom


class SharedPrivateProjector(nn.Module):
    def __init__(self, hidden_dim, dropout=0.35):
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.private = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.shared(x), self.private(x)


class AudioGuidedResidualMixer(nn.Module):
    """Inject audio context into other modality tokens with learnable gates."""

    def __init__(self, hidden_dim, audio_index=1, dropout=0.35):
        super().__init__()
        self.audio_index = int(audio_index)

        mid = max(32, hidden_dim // 2)
        self.gate_other = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 1),
        )
        self.gate_audio = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 1),
        )

        self.audio_to_other = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.other_to_audio = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.norm_other = nn.LayerNorm(hidden_dim)
        self.norm_audio = nn.LayerNorm(hidden_dim)

    def _gate(self, target, source, gate_net):
        feat = torch.cat([target, source, torch.abs(target - source)], dim=-1)
        return torch.sigmoid(gate_net(feat))

    def forward(self, tokens, token_mask=None):
        # tokens: (B, M, D)
        bsz, num_mod, _ = tokens.shape
        if token_mask is None:
            token_mask = torch.ones(bsz, num_mod, device=tokens.device, dtype=torch.bool)
        if num_mod <= 1:
            gates = token_mask.to(dtype=tokens.dtype)
            return tokens * token_mask.unsqueeze(-1), gates

        audio = tokens[:, self.audio_index]
        audio_valid = token_mask[:, self.audio_index]
        mixed = tokens.clone() * token_mask.unsqueeze(-1)
        gates = torch.zeros(bsz, num_mod, device=tokens.device, dtype=tokens.dtype)

        updated_non_audio = []
        updated_masks = []
        for idx in range(num_mod):
            if idx == self.audio_index:
                continue
            target = mixed[:, idx]
            target_valid = token_mask[:, idx]
            pair_valid = target_valid & audio_valid

            updated = target.clone()
            if pair_valid.any():
                g_valid = self._gate(target[pair_valid], audio[pair_valid], self.gate_other)
                updated[pair_valid] = self.norm_other(
                    target[pair_valid] + g_valid * self.audio_to_other(audio[pair_valid])
                )
                gates[pair_valid, idx] = g_valid.squeeze(-1)

            updated = updated * target_valid.unsqueeze(-1)
            mixed[:, idx] = updated
            updated_non_audio.append(updated)
            updated_masks.append(target_valid)

        if updated_non_audio:
            other_tokens = torch.stack(updated_non_audio, dim=1)
            other_mask = torch.stack(updated_masks, dim=1)
            has_other = other_mask.any(dim=1)
            audio_update_mask = audio_valid & has_other

            mixed_audio = audio.clone() * audio_valid.unsqueeze(-1)
            if audio_update_mask.any():
                others_mean = _masked_mean(other_tokens, other_mask)
                g_audio = self._gate(
                    audio[audio_update_mask],
                    others_mean[audio_update_mask],
                    self.gate_audio,
                )
                mixed_audio[audio_update_mask] = self.norm_audio(
                    audio[audio_update_mask]
                    + g_audio * self.other_to_audio(others_mean[audio_update_mask])
                )
                gates[audio_update_mask, self.audio_index] = g_audio.squeeze(-1)
            mixed[:, self.audio_index] = mixed_audio
        else:
            mixed[:, self.audio_index] = audio * audio_valid.unsqueeze(-1)

        mixed = mixed * token_mask.unsqueeze(-1)
        return mixed, gates


class PromptTokenFusion(nn.Module):
    """Concatenate modality tokens with learnable prompts then self-attend."""

    def __init__(self, hidden_dim, num_modalities=4, num_layers=1, dropout=0.35):
        super().__init__()
        self.num_modalities = int(num_modalities)
        self.prompt = nn.Parameter(torch.randn(1, self.num_modalities, hidden_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(num_layers)))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens, token_mask=None):
        # tokens: (B, M, D), M should equal num_modalities
        bsz, num_mod, _ = tokens.shape
        if num_mod != self.num_modalities:
            raise ValueError(f"PromptTokenFusion expects {self.num_modalities} modalities, got {num_mod}")
        if token_mask is None:
            token_mask = torch.ones(bsz, num_mod, device=tokens.device, dtype=torch.bool)

        prompt_tokens = self.prompt.expand(bsz, -1, -1)
        joint = torch.cat([tokens, prompt_tokens], dim=1)
        prompt_mask = torch.zeros(bsz, num_mod, device=tokens.device, dtype=torch.bool)
        key_padding_mask = torch.cat([~token_mask, prompt_mask], dim=1)
        joint = self.encoder(joint, src_key_padding_mask=key_padding_mask)

        content = self.norm(joint[:, :num_mod]) * token_mask.unsqueeze(-1)
        prompts = self.norm(joint[:, num_mod:])
        return content, prompts


class RelationGraphFusion(nn.Module):
    """Learned directed relation graph between modality tokens."""

    def __init__(self, hidden_dim, dropout=0.35):
        super().__init__()
        mid = max(32, hidden_dim // 2)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 1),
        )
        self.msg_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens, token_mask=None):
        # tokens: (B, M, D)
        bsz, num_mod, _ = tokens.shape
        if token_mask is None:
            token_mask = torch.ones(bsz, num_mod, device=tokens.device, dtype=torch.bool)
        if num_mod <= 1:
            eye = torch.eye(num_mod, device=tokens.device, dtype=tokens.dtype)
            return tokens * token_mask.unsqueeze(-1), eye.unsqueeze(0).expand(bsz, -1, -1)

        tgt = tokens.unsqueeze(2).expand(-1, num_mod, num_mod, -1)
        src = tokens.unsqueeze(1).expand(-1, num_mod, num_mod, -1)
        pair_feat = torch.cat([tgt, src, torch.abs(tgt - src)], dim=-1)

        logits = self.edge_mlp(pair_feat).squeeze(-1)
        eye = torch.eye(num_mod, device=tokens.device, dtype=torch.bool).unsqueeze(0)
        valid_pair = token_mask.unsqueeze(2) & token_mask.unsqueeze(1)
        valid_pair = valid_pair & ~eye
        logits = logits.masked_fill(~valid_pair, -1e4)

        edge_weights = torch.sigmoid(logits) * valid_pair.to(dtype=tokens.dtype)
        edge_weights = edge_weights / edge_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        msg_src = self.msg_linear(tokens)
        messages = torch.matmul(edge_weights, msg_src)
        fused = self.out_norm(tokens + messages) * token_mask.unsqueeze(-1)
        return fused, edge_weights


class LowRankMultiplicativeFusion(nn.Module):
    """Generalized LMF over modality-private tokens."""

    def __init__(self, hidden_dim, num_modalities=4, rank=10, dropout=0.35):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_modalities = int(num_modalities)
        self.rank = int(rank)

        self.factors = nn.ParameterList(
            [
                nn.Parameter(torch.empty(self.rank, self.hidden_dim + 1, self.hidden_dim))
                for _ in range(self.num_modalities)
            ]
        )
        self.rank_weights = nn.Parameter(torch.empty(self.rank))
        self.bias = nn.Parameter(torch.zeros(self.hidden_dim))

        self.out = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.hidden_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for factor in self.factors:
            nn.init.xavier_normal_(factor)
        nn.init.normal_(self.rank_weights, mean=0.0, std=0.02)

    def forward(self, private_tokens, token_mask=None):
        # private_tokens: (B, M, D)
        bsz, num_mod, hidden_dim = private_tokens.shape
        if num_mod != self.num_modalities:
            raise ValueError(f"LMF expects {self.num_modalities} modalities, got {num_mod}")
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"LMF hidden mismatch: expected {self.hidden_dim}, got {hidden_dim}")
        if token_mask is None:
            token_mask = torch.ones(bsz, num_mod, device=private_tokens.device, dtype=torch.bool)

        one = torch.ones(bsz, 1, device=private_tokens.device, dtype=private_tokens.dtype)

        rank_tensor = None
        for m_idx in range(self.num_modalities):
            x = private_tokens[:, m_idx]
            x = torch.cat([one, x], dim=1)
            z = torch.einsum("bd,rdh->brh", x, self.factors[m_idx])
            z = torch.where(
                token_mask[:, m_idx].view(bsz, 1, 1),
                z,
                torch.ones_like(z),
            )
            rank_tensor = z if rank_tensor is None else (rank_tensor * z)

        fused = torch.einsum("brh,r->bh", rank_tensor, self.rank_weights) + self.bias
        fused = self.out(fused)
        return fused, rank_tensor


class BlendEmoBackbone(nn.Module):
    """V40-compatible heads with 4-modality VideoMAE design."""

    MODALITY_NAMES = ("hicmae", "wavlm", "openface", "videomae")

    def __init__(
        self,
        num_set_classes,
        hicmae_dim=2048,
        wavlm_dim=1024,
        openface_dim=11,
        videomae_dim=1408,
        hidden_dim=192,
        dropout=0.35,
        lmf_rank=10,
        prompt_layers=1,
        enabled_modalities=None,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_modalities = len(self.MODALITY_NAMES)
        self.audio_index = 1
        self.enabled_modalities = _normalize_enabled_modalities(
            enabled_modalities,
            self.MODALITY_NAMES,
        )
        enabled_mask = torch.tensor(
            [name in self.enabled_modalities for name in self.MODALITY_NAMES],
            dtype=torch.bool,
        )
        self.register_buffer("enabled_modalities_mask", enabled_mask, persistent=True)

        # Temporal encoders per modality.
        self.hicmae_encoder = LightEncoder(hicmae_dim, hidden_dim, dropout)
        self.wavlm_encoder = SeqEncoder(wavlm_dim, hidden_dim, dropout)
        self.openface_encoder = SeqEncoder(openface_dim, hidden_dim, dropout)
        self.videomae_encoder = LightEncoder(videomae_dim, hidden_dim, dropout)

        self.hicmae_proj = SharedPrivateProjector(hidden_dim, dropout)
        self.wavlm_proj = SharedPrivateProjector(hidden_dim, dropout)
        self.openface_proj = SharedPrivateProjector(hidden_dim, dropout)
        self.videomae_proj = SharedPrivateProjector(hidden_dim, dropout)

        self.modality_embed = nn.Parameter(torch.randn(1, self.num_modalities, hidden_dim) * 0.02)

        self.audio_mixer = AudioGuidedResidualMixer(
            hidden_dim=hidden_dim,
            audio_index=self.audio_index,
            dropout=dropout,
        )
        self.prompt_fusion = PromptTokenFusion(
            hidden_dim=hidden_dim,
            num_modalities=self.num_modalities,
            num_layers=prompt_layers,
            dropout=dropout,
        )
        self.relation_fusion = RelationGraphFusion(hidden_dim, dropout)

        self.router = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.additive_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * (self.num_modalities + 1)),
            nn.Linear(hidden_dim * (self.num_modalities + 1), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        self.lmf = LowRankMultiplicativeFusion(
            hidden_dim=hidden_dim,
            num_modalities=self.num_modalities,
            rank=lmf_rank,
            dropout=dropout,
        )

        self.delta_emotion = nn.Linear(hidden_dim, 6)
        self.delta_mix = nn.Linear(hidden_dim, 1)
        self.delta_salience = nn.Linear(hidden_dim, 1)

        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3 + 3),
            nn.Linear(hidden_dim * 3 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

        self.emotion_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 6),
        )
        self.mix_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )
        self.salience_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3 + 3),
            nn.Linear(hidden_dim * 3 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.set_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_set_classes),
        )
        self.ratio_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + 1),
            nn.Linear(hidden_dim + 1, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 3),
        )

    def _uncertainty_features(self, emotion_logits, mix_logit):
        with torch.no_grad():
            probs = torch.softmax(emotion_logits, dim=-1)
            top2 = torch.topk(probs, 2, dim=-1).values
            gap = (top2[:, 0] - top2[:, 1]).unsqueeze(-1)

            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1, keepdim=True)
            entropy = entropy / math.log(float(probs.size(-1)))

            mix_prob = torch.sigmoid(mix_logit)
            mix_unc = 1.0 - 2.0 * (mix_prob - 0.5).abs()
            mix_unc = mix_unc.clamp(0.0, 1.0)

        return torch.cat([entropy, gap, mix_unc], dim=-1).detach()

    def _encode_modalities(
        self,
        hicmae,
        wavlm,
        openface,
        videomae,
        hicmae_mask=None,
        wavlm_mask=None,
        openface_mask=None,
        videomae_mask=None,
    ):
        bsz = hicmae.size(0)

        def encode_or_zero(enabled, encoder, projector, x, mask):
            if enabled:
                feat = encoder(x, mask)
                return projector(feat)
            zero = x.new_zeros(bsz, self.hidden_dim)
            return zero, zero

        s_v, p_v = encode_or_zero(
            bool(self.enabled_modalities_mask[0].item()),
            self.hicmae_encoder,
            self.hicmae_proj,
            hicmae,
            hicmae_mask,
        )
        s_a, p_a = encode_or_zero(
            bool(self.enabled_modalities_mask[1].item()),
            self.wavlm_encoder,
            self.wavlm_proj,
            wavlm,
            wavlm_mask,
        )
        s_f, p_f = encode_or_zero(
            bool(self.enabled_modalities_mask[2].item()),
            self.openface_encoder,
            self.openface_proj,
            openface,
            openface_mask,
        )
        s_m, p_m = encode_or_zero(
            bool(self.enabled_modalities_mask[3].item()),
            self.videomae_encoder,
            self.videomae_proj,
            videomae,
            videomae_mask,
        )

        shared_tokens = torch.stack([s_v, s_a, s_f, s_m], dim=1) + self.modality_embed
        private_tokens = torch.stack([p_v, p_a, p_f, p_m], dim=1)
        token_mask = self.enabled_modalities_mask.unsqueeze(0).expand(bsz, -1)
        shared_tokens = shared_tokens * token_mask.unsqueeze(-1)
        private_tokens = private_tokens * token_mask.unsqueeze(-1)
        return shared_tokens, private_tokens, token_mask

    def forward(
        self,
        hicmae,
        wavlm,
        openface,
        videomae,
        hicmae_mask=None,
        wavlm_mask=None,
        openface_mask=None,
        videomae_mask=None,
    ):
        shared_tokens, private_tokens, token_mask = self._encode_modalities(
            hicmae=hicmae,
            wavlm=wavlm,
            openface=openface,
            videomae=videomae,
            hicmae_mask=hicmae_mask,
            wavlm_mask=wavlm_mask,
            openface_mask=openface_mask,
            videomae_mask=videomae_mask,
        )

        mixed_tokens, audio_gates = self.audio_mixer(shared_tokens, token_mask)
        prompt_tokens, prompt_state = self.prompt_fusion(mixed_tokens, token_mask)
        rel_tokens, relation_weights = self.relation_fusion(prompt_tokens, token_mask)

        router_logits = self.router(rel_tokens).squeeze(-1)
        router_logits = router_logits.masked_fill(~token_mask, -1e4)
        router_weights = torch.softmax(router_logits, dim=1)
        router_pool = torch.sum(rel_tokens * router_weights.unsqueeze(-1), dim=1)

        z_add = self.additive_fusion(torch.cat([rel_tokens.flatten(1), router_pool], dim=-1))
        z_mul, rank_features = self.lmf(private_tokens, token_mask)

        base_emotion = self.emotion_head(z_add)
        base_mix = self.mix_head(z_add)

        private_global = _masked_mean(private_tokens, token_mask)
        unc0 = self._uncertainty_features(base_emotion, base_mix)
        base_salience = self.salience_head(torch.cat([z_add, private_global, z_mul, unc0], dim=-1))

        gate_in = torch.cat([z_add, z_mul, router_pool, unc0], dim=-1)
        gates = torch.sigmoid(self.gate(gate_in))

        emotion_logits = base_emotion + gates[:, 0:1] * self.delta_emotion(z_mul)
        mix_logit = base_mix + gates[:, 1:2] * self.delta_mix(z_mul)

        unc1 = self._uncertainty_features(emotion_logits, mix_logit)
        salience_delta = self.delta_salience(z_mul)
        salience_logit = (
            self.salience_head(torch.cat([z_add, private_global, z_mul, unc1], dim=-1))
            + gates[:, 2:3] * salience_delta
        )

        set_logits = self.set_head(z_add)
        ratio_logits = self.ratio_head(torch.cat([z_add, torch.sigmoid(mix_logit).detach()], dim=-1))

        fused_feature = z_add + 0.5 * z_mul + 0.5 * router_pool

        return {
            "emotion_logits": emotion_logits,
            "mix_logit": mix_logit,
            "salience_logit": salience_logit,
            "set_logits": set_logits,
            "ratio_logits": ratio_logits,
            "relation_weights": relation_weights,
            "router_weights": router_weights,
            "audio_gates": audio_gates,
            "shared_tokens": shared_tokens,
            "private_tokens": private_tokens,
            "token_mask": token_mask,
            "prompt_tokens": prompt_tokens,
            "prompt_state": prompt_state,
            "rank_features": rank_features,
            "fused_feature": fused_feature,
            "z_add": z_add,
            "z_mul": z_mul,
            "uncertainty": unc1,
        }


if __name__ == "__main__":
    model = BlendEmoBackbone(num_set_classes=16)
    out = model(
        hicmae=torch.randn(2, 6, 2048),
        wavlm=torch.randn(2, 180, 1024),
        openface=torch.randn(2, 180, 11),
        videomae=torch.randn(2, 18, 1408),
    )
    for key, value in out.items():
        if torch.is_tensor(value):
            print(key, tuple(value.shape))
    print(f"BlendEmo params: {sum(p.numel() for p in model.parameters()):,}")
