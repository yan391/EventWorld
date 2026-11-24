"""Horizon-conditioned forecasting and geometry-guided fusion."""
from __future__ import annotations

import torch
from torch import nn


class HorizonPredictor(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.physical_time = nn.Sequential(
            nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1024)
        )

    def forward(self, noisy: torch.Tensor, history: torch.Tensor, horizon_ms: int) -> torch.Tensor:
        patch_height, patch_width = 18, 36
        n, channels = 653, 1024
        hidden = noisy.reshape(1, n, channels)
        condition = history.reshape(1, 3 * n, channels)

        history_ms = torch.as_tensor([[-100, -50, 0]], device=noisy.device, dtype=torch.float32)
        target_ms = torch.as_tensor([[horizon_ms]], device=noisy.device, dtype=torch.float32)
        times = torch.cat((history_ms, target_ms), dim=1).float() / 50 + 2
        ids = torch.zeros((1, 4, n, 3), device=noisy.device, dtype=torch.float32)
        ids[..., 0] = times[:, :, None]
        y, x = torch.meshgrid(
            torch.arange(patch_height, device=noisy.device),
            torch.arange(patch_width, device=noisy.device),
            indexing="ij",
        )
        ids[:, :, 5:, 1] = y.reshape(1, 1, -1)
        ids[:, :, 5:, 2] = x.reshape(1, 1, -1)
        rotary = self.base.pos_embed(ids.reshape(1, 4 * n, 3))
        schedule_index = torch.zeros(1, device=noisy.device, dtype=torch.long)
        noise_time = self.base.timesteps_per_stage[schedule_index].to(hidden.dtype)
        temb = self.base.time_embed(noise_time)
        physical = self.base.time_embed.time_proj(target_ms[:, 0].float())
        temb = temb + self.physical_time(physical.to(self.physical_time[0].weight.dtype))

        total = 4 * n
        mask = [torch.ones((1, 1, total, total), device=noisy.device, dtype=torch.bool)]
        for block in self.base.transformer_blocks:
            condition, hidden = block(
                hidden_states=hidden, encoder_hidden_states=condition, encoder_attention_mask=None,
                temb=temb, attention_mask=mask, hidden_length=[n], image_rotary_emb=rotary,
            )
        hidden = torch.cat((condition, hidden), dim=1)
        for block in self.base.single_transformer_blocks:
            hidden = block(
                hidden_states=hidden, encoder_attention_mask=None, temb=temb, attention_mask=mask,
                hidden_length=[total], image_rotary_emb=rotary,
            )
        hidden = self.base.norm_out(hidden[:, 3 * n:], temb, hidden_length=[n])
        return self.base.proj_out(hidden).reshape(1, 1, n, channels).float()


class GeometryGuidedMixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.ModuleList(nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128))
                                     for width in (96, 192, 384))
        self.geometry = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 128))
        self.time = nn.Sequential(nn.Linear(2, 128), nn.SiLU(), nn.Linear(128, 128))
        self.score = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
        self.scale_bias = nn.Parameter(torch.zeros(3))
        self.output = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 1024))

    def forward(self, history: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        expanded = (motion, motion.repeat(1, 1, 1, 2), motion.repeat(1, 1, 1, 4))
        values = torch.stack([layer(tensor.float()) for layer, tensor in zip(self.project, expanded)], dim=-2)
        geometry = self.geometry(history[:, :, 5:].float()).unsqueeze(-2)
        offset = self.time(history.new_zeros(2, dtype=torch.float32)).reshape(1, 1, 1, 1, 128)
        weights = torch.softmax(self.score(torch.tanh(values + geometry + offset)).squeeze(-1)
                                + self.scale_bias, dim=-1)
        return self.output((weights.unsqueeze(-1) * values).sum(-2)).float()


class ForecastCore(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.direct = HorizonPredictor(base)
        self.additive = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 1024))
        self.single = GeometryGuidedMixture()

    def condition(self, history: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        if history.shape != (1, 3, 653, 1024) or motion.shape != (1, 3, 648, 96):
            raise ValueError("Unexpected geometry history or TESPEC grid")
        with torch.autocast("cuda", enabled=False):
            patches = history[:, :, 5:].float() + self.additive(motion.float())
            patches = patches + self.single(history, motion)
            return torch.cat((history[:, :, :5].float(), patches), dim=2)

    def endpoint(self, noise: torch.Tensor, fused_history: torch.Tensor, horizon_ms: int) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            clean = self.direct(noise, fused_history, horizon_ms)
        return noise.float() + (clean - noise.float())
