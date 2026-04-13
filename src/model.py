#!/usr/bin/env python3
"""BlendEmo model with structured set prediction and pair-conditioned ratio prediction.

Design:
1. Use a four-modality shared backbone.
2. Preserve the strong structured set head for presence prediction.
3. Use a pair-conditioned ratio predictor for mixed-emotion salience.
4. Keep fixed structured decoding at inference time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from labels import LABEL_TO_INDEX
from backbone import BlendEmoBackbone, _masked_mean


class BlendEmoModel(BlendEmoBackbone):
    """BlendEmo backbone plus pair-conditioned ratio head."""

    def __init__(
        self,
        set_classes,
        hicmae_dim=2048,
        wavlm_dim=1024,
        openface_dim=11,
        videomae_dim=1408,
        hidden_dim=192,
        dropout=0.35,
        lmf_rank=10,
        prompt_layers=1,
        enabled_modalities=None,
        ratio_mode="pair_conditioned",
        ratio_context_mode="full",
        disable_pair_embedding_branch=False,
        disable_pair_interaction=False,
        disable_emotion_prior=False,
        disable_setmix_prior=False,
        disable_audio_mixer=False,
        disable_prompt_fusion=False,
        disable_relation_fusion=False,
        router_pool_mode="router",
        disable_lmf_branch=False,
    ):
        self.set_classes = tuple(set_classes)
        self.num_single_classes = sum(1 for cls in self.set_classes if cls[0] == "single")
        self.mix_classes = tuple(cls for cls in self.set_classes if cls[0] == "mix")
        self.num_pair_classes = len(self.mix_classes)
        if self.num_single_classes <= 0 or self.num_pair_classes <= 0:
            raise ValueError("BlendEmo expects both single and mix classes to be present")
        if ratio_mode not in {"pair_conditioned", "global_v44"}:
            raise ValueError(f"Unsupported ratio_mode: {ratio_mode}")
        if ratio_context_mode not in {"full", "z_add_only"}:
            raise ValueError(f"Unsupported ratio_context_mode: {ratio_context_mode}")
        if router_pool_mode not in {"router", "mean"}:
            raise ValueError(f"Unsupported router_pool_mode: {router_pool_mode}")

        super().__init__(
            num_set_classes=len(self.set_classes),
            hicmae_dim=hicmae_dim,
            wavlm_dim=wavlm_dim,
            openface_dim=openface_dim,
            videomae_dim=videomae_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            lmf_rank=lmf_rank,
            prompt_layers=prompt_layers,
            enabled_modalities=enabled_modalities,
        )
        self.ratio_mode = str(ratio_mode)
        self.ratio_context_mode = str(ratio_context_mode)
        self.disable_pair_embedding_branch = bool(disable_pair_embedding_branch)
        self.disable_pair_interaction = bool(disable_pair_interaction)
        self.disable_emotion_prior = bool(disable_emotion_prior)
        self.disable_setmix_prior = bool(disable_setmix_prior)
        self.disable_audio_mixer = bool(disable_audio_mixer)
        self.disable_prompt_fusion = bool(disable_prompt_fusion)
        self.disable_relation_fusion = bool(disable_relation_fusion)
        self.router_pool_mode = str(router_pool_mode)
        self.disable_lmf_branch = bool(disable_lmf_branch)

        pair_emotion_index = []
        for cls in self.mix_classes:
            pair_emotion_index.append([LABEL_TO_INDEX[cls[1]], LABEL_TO_INDEX[cls[2]]])
        self.register_buffer(
            "pair_emotion_index",
            torch.tensor(pair_emotion_index, dtype=torch.long),
            persistent=True,
        )

        self.pair_embed = nn.Embedding(self.num_pair_classes, hidden_dim)
        self.global_ratio_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + 1),
            nn.Linear(hidden_dim + 1, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 3),
        )
        self.ratio_context = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3 + 3),
            nn.Linear(hidden_dim * 3 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.ratio_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3 + 8),
            nn.Linear(hidden_dim * 3 + 8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def _build_pair_conditioned_ratio_logits(self, fused_feature, z_add, z_mul, emotion_logits, mix_logit, set_logits):
        if self.ratio_mode == "global_v44":
            global_logits = self.global_ratio_head(
                torch.cat([z_add, torch.sigmoid(mix_logit).detach()], dim=-1)
            )
            return global_logits.unsqueeze(1).expand(-1, self.num_pair_classes, -1)

        if self.ratio_context_mode == "full":
            ratio_context_in = torch.cat(
                [
                    fused_feature,
                    z_add,
                    z_mul,
                    self._uncertainty_features(emotion_logits, mix_logit),
                ],
                dim=-1,
            )
        else:
            zeros_hidden = torch.zeros_like(z_add)
            zeros_unc = torch.zeros(z_add.size(0), 3, device=z_add.device, dtype=z_add.dtype)
            ratio_context_in = torch.cat([zeros_hidden, z_add, zeros_hidden, zeros_unc], dim=-1)

        ratio_context = self.ratio_context(
            ratio_context_in
        )
        context = ratio_context.unsqueeze(1).expand(-1, self.num_pair_classes, -1)
        pair_embed = self.pair_embed.weight.unsqueeze(0).expand(fused_feature.size(0), -1, -1)

        if self.disable_pair_embedding_branch:
            pair_embed_feat = torch.zeros_like(pair_embed)
            interaction = torch.zeros_like(pair_embed)
        else:
            pair_embed_feat = pair_embed
            interaction = context * pair_embed
            if self.disable_pair_interaction:
                interaction = torch.zeros_like(interaction)

        mix_set_logits = set_logits[:, self.num_single_classes :]
        mix_set_prob = torch.softmax(mix_set_logits, dim=-1)
        mix_prob = torch.sigmoid(mix_logit).expand(-1, self.num_pair_classes)

        emo_probs = torch.softmax(emotion_logits, dim=-1)
        idx1 = self.pair_emotion_index[:, 0]
        idx2 = self.pair_emotion_index[:, 1]
        e1_prob = emo_probs[:, idx1]
        e2_prob = emo_probs[:, idx2]
        e1_logit = emotion_logits[:, idx1]
        e2_logit = emotion_logits[:, idx2]

        pair_gap = e1_prob - e2_prob
        pair_abs_gap = torch.abs(pair_gap)

        scalar_feat = torch.stack(
            [
                e1_prob,
                e2_prob,
                e1_prob + e2_prob,
                pair_gap,
                pair_abs_gap,
                e1_logit - e2_logit,
                mix_set_prob,
                mix_prob,
            ],
            dim=-1,
        )
        scalar_feat = scalar_feat.clone()
        if self.disable_emotion_prior:
            scalar_feat[..., :6] = 0.0
        if self.disable_setmix_prior:
            scalar_feat[..., 6:] = 0.0

        ratio_in = torch.cat([context, pair_embed_feat, interaction, scalar_feat], dim=-1)
        return self.ratio_head(ratio_in)

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

        if self.disable_audio_mixer:
            mixed_tokens = shared_tokens
            audio_gates = torch.zeros_like(token_mask, dtype=shared_tokens.dtype)
        else:
            mixed_tokens, audio_gates = self.audio_mixer(shared_tokens, token_mask)

        if self.disable_prompt_fusion:
            prompt_tokens = mixed_tokens
            prompt_state = torch.zeros_like(mixed_tokens)
        else:
            prompt_tokens, prompt_state = self.prompt_fusion(mixed_tokens, token_mask)

        if self.disable_relation_fusion:
            rel_tokens = prompt_tokens
            eye = torch.eye(
                prompt_tokens.size(1),
                device=prompt_tokens.device,
                dtype=prompt_tokens.dtype,
            )
            relation_weights = eye.unsqueeze(0).expand(prompt_tokens.size(0), -1, -1).clone()
        else:
            rel_tokens, relation_weights = self.relation_fusion(prompt_tokens, token_mask)

        if self.router_pool_mode == "mean":
            router_weights = token_mask.to(dtype=rel_tokens.dtype)
            router_weights = router_weights / router_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            router_logits = self.router(rel_tokens).squeeze(-1)
            router_logits = router_logits.masked_fill(~token_mask, -1e4)
            router_weights = torch.softmax(router_logits, dim=1)
        router_pool = torch.sum(rel_tokens * router_weights.unsqueeze(-1), dim=1)

        z_add = self.additive_fusion(torch.cat([rel_tokens.flatten(1), router_pool], dim=-1))
        if self.disable_lmf_branch:
            z_mul = torch.zeros_like(z_add)
            rank_features = private_tokens.new_zeros(
                private_tokens.size(0),
                self.lmf.rank,
                self.hidden_dim,
            )
        else:
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
        fused_feature = z_add + 0.5 * z_mul + 0.5 * router_pool
        ratio_logits = self._build_pair_conditioned_ratio_logits(
            fused_feature=fused_feature,
            z_add=z_add,
            z_mul=z_mul,
            emotion_logits=emotion_logits,
            mix_logit=mix_logit,
            set_logits=set_logits,
        )

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
    set_classes = [
        ("single", "ang"),
        ("single", "disg"),
        ("single", "fea"),
        ("single", "hap"),
        ("single", "neu"),
        ("single", "sad"),
        ("mix", "ang", "disg"),
        ("mix", "ang", "fea"),
    ]
    model = BlendEmoModel(set_classes=set_classes)
    out = model(
        hicmae=torch.randn(2, 6, 2048),
        wavlm=torch.randn(2, 180, 1024),
        openface=torch.randn(2, 180, 11),
        videomae=torch.randn(2, 18, 1408),
    )
    for key, value in out.items():
        if torch.is_tensor(value):
            print(key, tuple(value.shape))
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
