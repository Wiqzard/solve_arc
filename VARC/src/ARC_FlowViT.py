from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    create_block_mask = None
    flex_attention = None
    FLEX_ATTENTION_AVAILABLE = False


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


class FramewiseSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        attention_backend: str,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if attention_backend not in {"auto", "flex", "sdpa"}:
            raise ValueError("attention_backend must be one of: auto, flex, sdpa")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.attention_backend = attention_backend

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

        self._flex_block_mask_cache: Dict[tuple[int, str], Any] = {}
        self._warned_flex_fallback = False

    def _resolve_backend(self, x: torch.Tensor) -> str:
        if self.attention_backend == "sdpa":
            return "sdpa"
        if self.attention_backend == "flex":
            if not FLEX_ATTENTION_AVAILABLE:
                raise RuntimeError("attention_backend=flex requested, but torch flex_attention is unavailable.")
            if not x.is_cuda:
                if not self._warned_flex_fallback:
                    print("Warning: flex attention requested on non-CUDA tensor; falling back to SDPA.")
                    self._warned_flex_fallback = True
                return "sdpa"
            return "flex"

        if FLEX_ATTENTION_AVAILABLE and x.is_cuda:
            return "flex"
        return "sdpa"

    def _get_dense_attn_mask(
        self,
        *,
        frame_index_per_token: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        causal = frame_index_per_token[:, None] >= frame_index_per_token[None, :]
        if key_padding_mask is None:
            return causal
        key_valid = (~key_padding_mask).unsqueeze(1).unsqueeze(1)
        return causal.unsqueeze(0).unsqueeze(0) & key_valid

    def _get_flex_block_mask(
        self,
        *,
        frame_index_per_token: torch.Tensor,
    ) -> Any:
        if create_block_mask is None:
            raise RuntimeError("torch flex_attention is unavailable.")
        seq_len = int(frame_index_per_token.numel())
        cache_key = (seq_len, str(frame_index_per_token.device))
        cached = self._flex_block_mask_cache.get(cache_key)
        if cached is not None:
            return cached

        frame_ids = frame_index_per_token

        def framewise_causal_mask(
            batch_idx: torch.Tensor,
            head_idx: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            del batch_idx, head_idx
            return frame_ids[q_idx] >= frame_ids[kv_idx]

        block_mask = create_block_mask(
            framewise_causal_mask,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=frame_index_per_token.device,
        )
        self._flex_block_mask_cache[cache_key] = block_mask
        return block_mask

    def forward(
        self,
        x: torch.Tensor,
        *,
        frame_index_per_token: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        backend = self._resolve_backend(x)
        if backend == "flex":
            block_mask = self._get_flex_block_mask(frame_index_per_token=frame_index_per_token)
            score_mod = None
            if key_padding_mask is not None:
                key_valid = (~key_padding_mask).to(device=x.device, dtype=torch.bool)
                min_value = torch.finfo(q.dtype).min

                def score_mod(
                    score: torch.Tensor,
                    batch_idx: torch.Tensor,
                    head_idx: torch.Tensor,
                    q_idx: torch.Tensor,
                    kv_idx: torch.Tensor,
                ) -> torch.Tensor:
                    del head_idx, q_idx
                    keep = key_valid[batch_idx, kv_idx]
                    return torch.where(keep, score, min_value)

            context = flex_attention(q, k, v, block_mask=block_mask, score_mod=score_mod)
        else:
            attn_mask = self._get_dense_attn_mask(
                frame_index_per_token=frame_index_per_token,
                key_padding_mask=key_padding_mask,
            )
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )

        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        context = self.proj(context)
        context = self.proj_dropout(context)

        if key_padding_mask is not None:
            context = context.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return context


class FramewiseTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attention_backend: str,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = FramewiseSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_backend=attention_backend,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        ff_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        frame_index_per_token: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            frame_index_per_token=frame_index_per_token,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.mlp(self.norm2(x))
        return x


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
        framewise_causal_attention: bool = False,
        attention_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.num_colors = num_colors
        self.max_frames = max_frames
        self.spatial_tokens = image_size * image_size
        self.embed_dim = embed_dim
        self.framewise_causal_attention = framewise_causal_attention
        self.attention_backend = attention_backend
        self.depth = depth

        self.input_proj = nn.Linear(num_colors, embed_dim)
        self.frame_embed = nn.Embedding(max_frames, embed_dim)
        self.spatial_embed = nn.Parameter(torch.zeros(1, self.spatial_tokens, embed_dim))
        token_frame_index = torch.arange(max_frames, dtype=torch.long).repeat_interleave(self.spatial_tokens)
        self.register_buffer("token_frame_index", token_frame_index, persistent=False)

        self.time_embed_base = SinusoidalTimeEmbedding(embed_dim)
        self.time_embed_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.SiLU(),
                    nn.Linear(embed_dim, embed_dim),
                )
                for _ in range(depth)
            ]
        )

        if framewise_causal_attention:
            self.encoder_layers = nn.ModuleList(
                [
                    FramewiseTransformerBlock(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_backend=attention_backend,
                    )
                    for _ in range(depth)
                ]
            )
        else:
            ff_dim = int(embed_dim * mlp_ratio)
            self.encoder_layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=num_heads,
                        dim_feedforward=ff_dim,
                        dropout=dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(depth)
                ]
            )
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

        tokens = tokens.reshape(batch_size, frames * self.spatial_tokens, self.embed_dim)
        frame_index_per_token = self.token_frame_index[: frames * self.spatial_tokens]

        key_padding_mask = None
        if frame_valid_mask is not None:
            if frame_valid_mask.shape != (batch_size, frames, height, width):
                raise ValueError("frame_valid_mask shape mismatch")
            key_padding_mask = ~frame_valid_mask.reshape(batch_size, frames * self.spatial_tokens).bool()

        base_time_embed = self.time_embed_base(frame_times.reshape(-1)).reshape(batch_size, frames, self.embed_dim)
        encoded = tokens
        for layer_idx, layer in enumerate(self.encoder_layers):
            layer_time_embed = self.time_embed_layers[layer_idx](
                base_time_embed.reshape(-1, self.embed_dim)
            ).reshape(batch_size, frames, self.embed_dim)
            encoded = encoded + layer_time_embed[:, frame_index_per_token, :]
            if self.framewise_causal_attention:
                encoded = layer(
                    encoded,
                    frame_index_per_token=frame_index_per_token,
                    key_padding_mask=key_padding_mask,
                )
            else:
                encoded = layer(encoded, src_key_padding_mask=key_padding_mask)
        encoded = self.norm(encoded)
        velocity = self.head(encoded)
        velocity = velocity.reshape(batch_size, frames, height, width, self.num_colors)
        return velocity
