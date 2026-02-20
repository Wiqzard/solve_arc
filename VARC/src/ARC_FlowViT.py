from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() != 1:
            raise ValueError("timesteps must be 1D")
        half_dim = self.dim // 2
        if half_dim == 0:
            return timesteps.unsqueeze(-1)
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=timesteps.device, dtype=timesteps.dtype) / max(half_dim - 1, 1)
        )
        angles = timesteps.unsqueeze(-1) * freq.unsqueeze(0)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if emb.size(-1) < self.dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class ARCFlowViT(nn.Module):
    """Frame-sequence ViT for flow matching on ARC episodes."""

    def __init__(
        self,
        *,
        image_size: int,
        num_colors: int,
        max_frames: int,
        embed_dim: int = 512,
        depth: int = 10,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.num_colors = num_colors
        self.max_frames = max_frames
        self.spatial_tokens = image_size * image_size
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(num_colors, embed_dim)
        self.frame_embed = nn.Embedding(max_frames, embed_dim)
        self.spatial_embed = nn.Parameter(torch.zeros(1, self.spatial_tokens, embed_dim))

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        ff_dim = int(embed_dim * mlp_ratio)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_colors)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.spatial_embed, std=0.02)
        nn.init.trunc_normal_(self.frame_embed.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        frame_times: torch.Tensor,
        *,
        frame_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_t.dim() != 5:
            raise ValueError("x_t must be (batch, frames, height, width, channels).")
        batch_size, frames, height, width, channels = x_t.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}, got {(height, width)}")
        if channels != self.num_colors:
            raise ValueError(f"Expected num_colors={self.num_colors}, got {channels}")
        if frames > self.max_frames:
            raise ValueError(f"frames={frames} exceeds max_frames={self.max_frames}")
        if frame_times.shape != (batch_size, frames):
            raise ValueError(f"frame_times must be {(batch_size, frames)}")

        tokens = x_t.reshape(batch_size, frames, self.spatial_tokens, self.num_colors)
        tokens = self.input_proj(tokens)

        frame_ids = torch.arange(frames, device=x_t.device, dtype=torch.long).view(1, frames, 1)
        tokens = tokens + self.frame_embed(frame_ids)
        tokens = tokens + self.spatial_embed[:, None, :, :]

        time_tokens = self.time_embed(frame_times.reshape(-1)).reshape(batch_size, frames, 1, self.embed_dim)
        tokens = tokens + time_tokens

        tokens = tokens.reshape(batch_size, frames * self.spatial_tokens, self.embed_dim)

        key_padding_mask = None
        if frame_valid_mask is not None:
            if frame_valid_mask.shape != (batch_size, frames, height, width):
                raise ValueError("frame_valid_mask shape mismatch")
            key_padding_mask = ~frame_valid_mask.reshape(batch_size, frames * self.spatial_tokens).bool()

        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        encoded = self.norm(encoded)
        velocity = self.head(encoded)
        velocity = velocity.reshape(batch_size, frames, height, width, self.num_colors)
        return velocity
