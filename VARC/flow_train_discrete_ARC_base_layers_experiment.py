from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.nn.parallel import DistributedDataParallel as DDP

from flow_train_discrete_ARC import (
    autocast_context,
    build_discrete_path,
    cleanup_distributed,
    cudagraph_step_begin_if_available,
    discrete_flow_matching_loss,
    evaluate_discrete_flow_loss,
    evaluate_last_frame_accuracy,
    get_loss_function,
    maybe_compile_model,
    one_hot_frames,
    parse_optional_bool,
    sample_frame_times,
    sample_xt_from_qt,
    save_checkpoint,
    set_seed,
    setup_distributed,
)
from src.ARC_FlowViT import ARCFlowViT
from src.ARC_context_flow_loader import build_flow_context_dataloaders

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable

try:
    import wandb
except ImportError:
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discrete flow matching experiment with frozen selected base layers, adapters, and optional LoRA. "
            "Designed for transfer from external pretrained checkpoints (for example, converted Hunyuan weights)."
        )
    )
    parser.add_argument("--data-root", type=str, default="raw_data/ARC-AGI")
    parser.add_argument("--train-split", type=str, default="training")
    parser.add_argument("--eval-split", type=str, default="evaluation")

    parser.add_argument(
        "--max-demos",
        "--num-demos",
        dest="max_demos",
        type=int,
        default=10,
        help="Maximum number of demonstration pairs (m) in the context.",
    )
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--num-colors", type=int, default=12)

    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--framewise-causal-attention",
        nargs="?",
        const=True,
        default=True,
        type=parse_optional_bool,
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="auto",
        choices=("auto", "flex", "sdpa"),
    )
    parser.add_argument(
        "--mask-pad-attention",
        default=True,
        nargs="?",
        const=True,
        type=parse_optional_bool,
    )
    parser.add_argument(
        "--mask-intra-frame-pad-attention",
        default=False,
        nargs="?",
        const=True,
        type=parse_optional_bool,
    )
    parser.add_argument(
        "--rope-3d",
        nargs="?",
        const=True,
        default=True,
        type=parse_optional_bool,
    )
    parser.add_argument("--rope-base", type=float, default=256.0)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=("cosine", "none"))
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every-steps", type=int, default=20)
    parser.add_argument("--verbose", action="store_true", default=False)

    parser.add_argument(
        "--compile",
        nargs="?",
        const=True,
        default=True,
        type=parse_optional_bool,
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        choices=("default", "reduce-overhead", "max-autotune"),
    )

    parser.add_argument(
        "--include-rearc",
        nargs="?",
        const=True,
        default=False,
        type=parse_optional_bool,
    )
    parser.add_argument("--rearc-path", type=str, default="raw_data/re_arc")
    parser.add_argument("--rearc-limit", type=int, default=-1)
    parser.add_argument(
        "--include-barc",
        nargs="?",
        const=True,
        default=False,
        type=parse_optional_bool,
    )
    parser.add_argument("--barc-path", type=str, default="raw_data/BARC")
    parser.add_argument("--barc-limit", type=int, default=-1)

    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--dist-backend", type=str, default="nccl", choices=("nccl", "gloo"))
    parser.add_argument("--dist-url", type=str, default="env://")
    parser.add_argument(
        "--bf16-autocast",
        nargs="?",
        const=True,
        default=True,
        type=parse_optional_bool,
    )

    parser.add_argument(
        "--flow-train-translation-aug",
        nargs="?",
        const=True,
        default=True,
        type=parse_optional_bool,
    )
    parser.add_argument(
        "--flow-train-resolution-aug",
        nargs="?",
        const=True,
        default=True,
        type=parse_optional_bool,
    )
    parser.add_argument(
        "--nested-dropout",
        default=True,
        nargs="?",
        const=True,
        type=parse_optional_bool,
    )

    parser.add_argument("--loss-on-target-only", action="store_true", default=False)
    parser.add_argument(
        "--loss-function",
        type=str,
        default="cross_entropy",
        choices=("cross_entropy", "generalized_kl"),
    )
    parser.add_argument("--discrete-rate", type=float, default=5.0)
    parser.add_argument(
        "--discrete-scheduler",
        type=str,
        default="cosine",
        choices=("exponential", "condot", "polynomial", "vp", "linear_vp", "cosine"),
    )
    parser.add_argument("--discrete-poly-n", type=float, default=2.0)
    parser.add_argument("--discrete-vp-beta-min", type=float, default=0.1)
    parser.add_argument("--discrete-vp-beta-max", type=float, default=20.0)
    parser.add_argument("--reverse-sampler", type=str, default="sample", choices=("sample", "argmax"))

    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=500,
        help="If > 0, run evaluation every N optimizer steps (disables epoch-based eval).",
    )
    parser.add_argument("--sample-steps", type=int, default=40)
    parser.add_argument("--train-time-discretization-steps", type=int, default=1000)

    parser.add_argument(
        "--save-path",
        type=str,
        default="saves/flow_context_vit_discrete_base_layers/checkpoint_last.pt",
    )
    parser.add_argument(
        "--best-save-path",
        type=str,
        default="saves/flow_context_vit_discrete_base_layers/checkpoint_best.pt",
    )

    parser.add_argument("--use-wandb", nargs="?", const=True, default=True, type=parse_optional_bool)
    parser.add_argument("--wandb-project", type=str, default="VisionARC")
    parser.add_argument("--wandb-run-name", type=str, default="flow-context-vit-discrete-base-layers")

    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default="",
        help="Path to source checkpoint (local). Supports full payload with model_state/state_dict or raw state dict.",
    )
    parser.add_argument(
        "--base-state-key",
        type=str,
        default="auto",
        choices=("auto", "model_state", "state_dict", "raw"),
        help="How to read state dict from --base-checkpoint.",
    )
    parser.add_argument(
        "--base-checkpoint-strip-prefix",
        type=str,
        default="",
        help="Optional prefix to strip from source checkpoint keys before matching.",
    )
    parser.add_argument(
        "--base-load-scope",
        type=str,
        default="all",
        choices=("all", "encoder_only"),
        help="all: load every matching parameter, encoder_only: load only encoder/time layers.",
    )

    parser.add_argument(
        "--layer-selection-strategy",
        type=str,
        default="gradient_taylor",
        choices=("manual", "gradient_taylor", "random"),
    )
    parser.add_argument(
        "--selected-layer-ids",
        type=str,
        default="",
        help="Comma-separated layer indices, used when --layer-selection-strategy=manual.",
    )
    parser.add_argument(
        "--max-base-layers",
        type=int,
        default=8,
        help="Maximum number of selected base layers used in the experiment.",
    )
    parser.add_argument(
        "--importance-batches",
        type=int,
        default=20,
        help="Number of mini-batches used to estimate layer importance when using gradient_taylor.",
    )
    parser.add_argument(
        "--importance-save-path",
        type=str,
        default="saves/flow_context_vit_discrete_base_layers/layer_importance.json",
    )

    parser.add_argument(
        "--freeze-selected-base-layers",
        nargs="?",
        const=True,
        default=True,
        type=parse_optional_bool,
        help="Keep selected base layers frozen during training.",
    )
    parser.add_argument("--lora-rank", type=int, default=0, help="0 disables LoRA.")
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-targets",
        type=str,
        default="attn.qkv,attn.proj,mlp.0,mlp.3",
        help="Comma-separated linear module paths inside each selected encoder layer.",
    )

    args = parser.parse_args()
    args.num_demos = args.max_demos
    if args.max_base_layers < 1:
        raise ValueError("--max-base-layers must be >= 1")
    return args


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")
        self.base = base_linear
        for param in self.base.parameters():
            param.requires_grad = False
        self.rank = rank
        self.scaling = alpha / float(rank)
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base_linear.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base_linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        return base_out + lora_out


def _split_csv(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_selected_layer_ids(text: str) -> List[int]:
    if not text.strip():
        return []
    out: List[int] = []
    for part in _split_csv(text):
        out.append(int(part))
    return out


def extract_state_dict(payload: Any, state_key: str) -> Dict[str, torch.Tensor]:
    if state_key == "raw":
        if not isinstance(payload, dict):
            raise ValueError("Expected raw checkpoint to be a state dict.")
        return payload
    if state_key == "model_state":
        if not isinstance(payload, dict) or "model_state" not in payload:
            raise ValueError("Checkpoint does not contain 'model_state'.")
        return payload["model_state"]
    if state_key == "state_dict":
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise ValueError("Checkpoint does not contain 'state_dict'.")
        return payload["state_dict"]

    if not isinstance(payload, dict):
        raise ValueError("Expected checkpoint payload to be a dict.")
    if "model_state" in payload and isinstance(payload["model_state"], dict):
        return payload["model_state"]
    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    return payload


def load_base_weights(
    model: nn.Module,
    *,
    checkpoint_path: str,
    state_key: str,
    strip_prefix: str,
    load_scope: str,
    is_main: bool,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "loaded": 0,
        "skipped_missing": 0,
        "skipped_shape": 0,
        "source_keys": 0,
    }
    if not checkpoint_path:
        if is_main:
            print("No --base-checkpoint provided, using random initialization for source model.")
        return info

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = extract_state_dict(checkpoint, state_key=state_key)
    model_state = model.state_dict()

    filtered: Dict[str, torch.Tensor] = {}
    for key, value in source_state.items():
        if not isinstance(value, torch.Tensor):
            continue
        mapped_key = key
        if mapped_key.startswith("module."):
            mapped_key = mapped_key[len("module.") :]
        if strip_prefix and mapped_key.startswith(strip_prefix):
            mapped_key = mapped_key[len(strip_prefix) :]
        if load_scope == "encoder_only" and not (
            mapped_key.startswith("encoder_layers.") or mapped_key.startswith("time_embed_layers.")
        ):
            continue
        info["source_keys"] += 1
        if mapped_key not in model_state:
            info["skipped_missing"] += 1
            continue
        if tuple(model_state[mapped_key].shape) != tuple(value.shape):
            info["skipped_shape"] += 1
            continue
        filtered[mapped_key] = value

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    info["loaded"] = len(filtered)
    info["unexpected_after_filter"] = len(unexpected)
    info["missing_after_filter"] = len(missing)
    if is_main:
        print(
            "Loaded base checkpoint",
            f"path={checkpoint_path}",
            f"matched={info['loaded']}",
            f"source_tensor_keys={info['source_keys']}",
            f"skipped_missing={info['skipped_missing']}",
            f"skipped_shape={info['skipped_shape']}",
        )
    return info


def get_submodule(root: nn.Module, dotted_path: str) -> nn.Module:
    module: nn.Module = root
    for part in dotted_path.split("."):
        module = getattr(module, part)
    return module


def set_submodule(root: nn.Module, dotted_path: str, value: nn.Module) -> None:
    if "." not in dotted_path:
        setattr(root, dotted_path, value)
        return
    parent_path, leaf = dotted_path.rsplit(".", 1)
    parent = get_submodule(root, parent_path)
    setattr(parent, leaf, value)


def freeze_selected_layers(model: ARCFlowViT, selected_layers: Sequence[int]) -> None:
    for layer_idx in selected_layers:
        for param in model.encoder_layers[layer_idx].parameters():
            param.requires_grad = False
        for param in model.time_embed_layers[layer_idx].parameters():
            param.requires_grad = False


def apply_lora_to_selected_layers(
    model: ARCFlowViT,
    *,
    selected_layers: Sequence[int],
    rank: int,
    alpha: float,
    dropout: float,
    target_paths: Sequence[str],
    is_main: bool,
) -> int:
    if rank <= 0:
        return 0

    replaced = 0
    for layer_idx in selected_layers:
        layer = model.encoder_layers[layer_idx]
        for path in target_paths:
            try:
                module = get_submodule(layer, path)
            except AttributeError:
                continue
            if not isinstance(module, nn.Linear):
                continue
            lora_module = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
            set_submodule(layer, path, lora_module)
            replaced += 1

    if is_main:
        print(
            f"Applied LoRA to {replaced} linear modules "
            f"(rank={rank}, alpha={alpha}, dropout={dropout})."
        )
    return replaced


def count_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return trainable, total


class ARCFlowViTSelectedLayersAdapter(ARCFlowViT):
    """Use selected encoder layers as frozen base stack with trainable adapters before/after."""

    def __init__(
        self,
        *,
        selected_layers: Sequence[int],
        image_size: int,
        patch_size: int,
        num_colors: int,
        max_frames: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        framewise_causal_attention: bool,
        mask_pad_tokens_in_attention: bool,
        mask_intra_frame_pad_tokens_in_attention: bool,
        attention_backend: str,
        rope_3d: bool,
        rope_base: float,
    ) -> None:
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_colors=num_colors,
            max_frames=max_frames,
            embed_dim=embed_dim,
            depth=depth,
            n_loops=1,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            framewise_causal_attention=framewise_causal_attention,
            mask_pad_tokens_in_attention=mask_pad_tokens_in_attention,
            mask_intra_frame_pad_tokens_in_attention=mask_intra_frame_pad_tokens_in_attention,
            attention_backend=attention_backend,
            rope_3d=rope_3d,
            rope_base=rope_base,
        )
        cleaned = sorted(set(int(idx) for idx in selected_layers))
        if not cleaned:
            raise ValueError("selected_layers cannot be empty")
        if cleaned[-1] >= depth or cleaned[0] < 0:
            raise ValueError(f"selected layer indices must be in [0, {depth - 1}]")
        self.selected_layers = tuple(cleaned)

        self.pre_adapter = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.post_adapter = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

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

        pixel_tokens = self.input_proj(x_t)
        patch_tokens = self.patch_embed(
            pixel_tokens.reshape(batch_size * frames, height, width, self.embed_dim).permute(0, 3, 1, 2)
        )
        tokens = patch_tokens.reshape(batch_size, frames, self.spatial_tokens, self.embed_dim)

        frame_ids = torch.arange(frames, device=x_t.device, dtype=torch.long).view(1, frames, 1)
        tokens = tokens + self.frame_embed(frame_ids)
        tokens = tokens + self.spatial_embed[:, None, :, :]

        tokens = tokens.reshape(batch_size, frames * self.spatial_tokens, self.embed_dim)
        frame_index_per_token = self.token_frame_index[: frames * self.spatial_tokens]
        y_index_per_token = self.token_y_index[: frames * self.spatial_tokens]
        x_index_per_token = self.token_x_index[: frames * self.spatial_tokens]

        key_padding_mask = None
        if frame_valid_mask is not None:
            if frame_valid_mask.shape != (batch_size, frames, height, width):
                raise ValueError("frame_valid_mask shape mismatch")
            frame_valid_tokens = frame_valid_mask.bool()
            frame_valid_tokens = frame_valid_tokens.reshape(
                batch_size,
                frames,
                self.grid_size,
                self.patch_size,
                self.grid_size,
                self.patch_size,
            )
            frame_valid_tokens = frame_valid_tokens.any(dim=3).any(dim=4)
            frame_valid_tokens = frame_valid_tokens.reshape(batch_size, frames, self.spatial_tokens)
            if self.mask_intra_frame_pad_tokens_in_attention:
                token_valid_for_attention = frame_valid_tokens
            elif self.mask_pad_tokens_in_attention:
                frame_is_active = frame_valid_tokens.any(dim=-1, keepdim=True)
                token_valid_for_attention = frame_is_active.expand(-1, -1, self.spatial_tokens)
            else:
                token_valid_for_attention = None
            if token_valid_for_attention is not None:
                key_padding_mask = ~token_valid_for_attention.reshape(batch_size, frames * self.spatial_tokens)

        base_time_embed = self.time_embed_base(frame_times.reshape(-1)).reshape(batch_size, frames, self.embed_dim)
        encoded = tokens + self.pre_adapter(tokens)

        for layer_idx in self.selected_layers:
            layer_time_embed = self.time_embed_layers[layer_idx](
                base_time_embed.reshape(-1, self.embed_dim)
            ).reshape(batch_size, frames, self.embed_dim)
            encoded = encoded + layer_time_embed[:, frame_index_per_token, :]
            layer = self.encoder_layers[layer_idx]
            if self.use_custom_attention_blocks:
                encoded = layer(
                    encoded,
                    frame_index_per_token=frame_index_per_token,
                    frame_count=frames,
                    spatial_tokens_per_frame=self.spatial_tokens,
                    y_index_per_token=y_index_per_token,
                    x_index_per_token=x_index_per_token,
                    key_padding_mask=key_padding_mask,
                )
            else:
                encoded = layer(encoded, src_key_padding_mask=key_padding_mask)

        encoded = encoded + self.post_adapter(encoded)
        encoded = self.norm(encoded)
        velocity = self.head(encoded)
        velocity = velocity.reshape(
            batch_size,
            frames,
            self.grid_size,
            self.grid_size,
            self.patch_size,
            self.patch_size,
            self.num_colors,
        )
        velocity = velocity.permute(0, 1, 2, 4, 3, 5, 6).reshape(batch_size, frames, height, width, self.num_colors)
        return velocity


@torch.no_grad()
def _broadcast_selected_mask(
    *,
    selected_layers: Sequence[int],
    depth: int,
    device: torch.device,
    distributed: bool,
) -> List[int]:
    mask = torch.zeros(depth, device=device, dtype=torch.bool)
    for idx in selected_layers:
        if 0 <= idx < depth:
            mask[idx] = True
    if distributed:
        dist.broadcast(mask, src=0)
    return torch.nonzero(mask, as_tuple=False).flatten().tolist()


def compute_layer_importance_gradient_taylor(
    *,
    model: ARCFlowViT,
    train_loader: Iterable[Dict[str, Any]],
    device: torch.device,
    num_colors: int,
    time_discretization_steps: int,
    path,
    loss_function: _Loss,
    target_only: bool,
    importance_batches: int,
    bf16_autocast: bool,
    distributed: bool,
) -> List[float]:
    if importance_batches <= 0:
        raise ValueError("importance_batches must be > 0")

    model.train()
    layer_scores = torch.zeros(len(model.encoder_layers), device=device, dtype=torch.float64)
    loader_iter = iter(train_loader)

    for _ in range(importance_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        frames = batch["frames"].to(device)
        frame_valid_mask = batch["frame_valid_mask"].to(device)
        target_frame_index = batch["target_frame_index"].to(device)
        target_valid_mask = batch["target_valid_mask"].to(device)

        batch_size, frame_count, _, _ = frames.shape
        frame_times = sample_frame_times(
            batch_size=batch_size,
            frames=frame_count,
            device=device,
            discretization_steps=time_discretization_steps,
        )
        x_t_tokens = sample_xt_from_qt(
            frames,
            frame_valid_mask,
            frame_times,
            num_colors=num_colors,
            path=path,
        )
        x_t = one_hot_frames(x_t_tokens, num_colors=num_colors)

        model.zero_grad(set_to_none=True)
        with autocast_context(device, bf16_autocast):
            logits = model(x_t, frame_times, frame_valid_mask=frame_valid_mask)
        loss = discrete_flow_matching_loss(
            logits.float(),
            frames,
            x_t_tokens,
            frame_valid_mask=frame_valid_mask,
            frame_times=frame_times,
            target_frame_index=target_frame_index,
            target_valid_mask=target_valid_mask,
            loss_function=loss_function,
            target_only=target_only,
        )
        loss.backward()

        for layer_idx, layer in enumerate(model.encoder_layers):
            score = 0.0
            for param in layer.parameters():
                if param.grad is None:
                    continue
                score += torch.sum((param.grad.detach() * param.detach()).abs()).double()
            layer_scores[layer_idx] += score

    if distributed:
        dist.all_reduce(layer_scores, op=dist.ReduceOp.SUM)

    denom = float(max(importance_batches, 1))
    layer_scores = layer_scores / denom
    return [float(v) for v in layer_scores.detach().cpu().tolist()]


def choose_selected_layers(
    *,
    model: ARCFlowViT,
    args: argparse.Namespace,
    train_loader: Iterable[Dict[str, Any]],
    path,
    loss_function: _Loss,
    device: torch.device,
    bf16_autocast: bool,
    distributed: bool,
    rank: int,
    is_main: bool,
) -> Tuple[List[int], Dict[str, float]]:
    max_layers = min(args.max_base_layers, len(model.encoder_layers))

    if args.layer_selection_strategy == "manual":
        selected = parse_selected_layer_ids(args.selected_layer_ids)
        if not selected:
            raise ValueError("Manual layer selection requires --selected-layer-ids.")
        if len(selected) > max_layers:
            selected = selected[:max_layers]
        importance = {str(i): float("nan") for i in range(len(model.encoder_layers))}
        if is_main:
            print(f"Using manual selected layers: {selected}")
    elif args.layer_selection_strategy == "random":
        rng = random.Random(args.seed)
        all_indices = list(range(len(model.encoder_layers)))
        rng.shuffle(all_indices)
        selected = sorted(all_indices[:max_layers])
        importance = {str(i): float("nan") for i in range(len(model.encoder_layers))}
        if is_main:
            print(f"Using random selected layers: {selected}")
    else:
        scores = compute_layer_importance_gradient_taylor(
            model=model,
            train_loader=train_loader,
            device=device,
            num_colors=args.num_colors,
            time_discretization_steps=args.train_time_discretization_steps,
            path=path,
            loss_function=loss_function,
            target_only=args.loss_on_target_only,
            importance_batches=args.importance_batches,
            bf16_autocast=bf16_autocast,
            distributed=distributed,
        )
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        selected = sorted(idx for idx, _ in ranked[:max_layers])
        importance = {str(idx): float(score) for idx, score in enumerate(scores)}
        if is_main:
            ranking_text = ", ".join([f"L{idx}:{score:.4e}" for idx, score in ranked])
            print(f"Layer importance ranking ({args.layer_selection_strategy}): {ranking_text}")
            print(f"Selected top-{max_layers} layers (execution order): {selected}")

    selected = [idx for idx in selected if 0 <= idx < len(model.encoder_layers)]
    if not selected:
        raise RuntimeError("No valid selected layers after filtering.")

    if distributed:
        if rank == 0:
            sync_selected = selected
        else:
            sync_selected = []
        selected = _broadcast_selected_mask(
            selected_layers=sync_selected,
            depth=len(model.encoder_layers),
            device=device,
            distributed=True,
        )

    return selected, importance


def run_evaluation(
    *,
    model: ARCFlowViT,
    eval_loader,
    fallback_loader,
    device: torch.device,
    args: argparse.Namespace,
    path,
    loss_function: _Loss,
    bf16_autocast: bool,
    show_progress: bool,
) -> Tuple[float, Dict[str, float]]:
    loader = eval_loader if eval_loader is not None else fallback_loader
    eval_loss = evaluate_discrete_flow_loss(
        model,
        loader,
        device=device,
        num_colors=args.num_colors,
        time_discretization_steps=args.train_time_discretization_steps,
        path=path,
        loss_function=loss_function,
        target_only=args.loss_on_target_only,
        show_progress=False,
        autocast_enabled=bf16_autocast,
    )
    eval_metrics, _ = evaluate_last_frame_accuracy(
        model,
        loader,
        device=device,
        num_colors=args.num_colors,
        sample_steps=args.sample_steps,
        path=path,
        reverse_sampler=args.reverse_sampler,
        collect_examples=0,
        show_progress=show_progress,
        autocast_enabled=bf16_autocast,
    )
    return eval_loss, eval_metrics


def train(args: argparse.Namespace) -> None:
    distributed, rank, local_rank, world_size, device = setup_distributed(args)
    is_main = rank == 0
    set_seed(args.seed + rank)

    bf16_autocast = bool(args.bf16_autocast and device.type == "cuda" and torch.cuda.is_bf16_supported())
    if args.bf16_autocast and is_main and not bf16_autocast:
        print("Warning: BF16 autocast requested but unavailable on this device. Falling back to fp32.")

    train_dataset, train_loader, eval_dataset, eval_loader, train_sampler, eval_sampler = build_flow_context_dataloaders(
        args,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    max_frames = 2 * args.num_demos + 2
    if args.image_size % args.patch_size != 0:
        raise ValueError(f"image-size ({args.image_size}) must be divisible by patch-size ({args.patch_size}).")
    if is_main and args.verbose:
        print(f"Train episodes: {len(train_dataset)}")
        if eval_dataset is not None:
            print(f"Eval episodes: {len(eval_dataset)}")

    donor_model = ARCFlowViT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_colors=args.num_colors,
        max_frames=max_frames,
        embed_dim=args.embed_dim,
        depth=args.depth,
        n_loops=1,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        framewise_causal_attention=args.framewise_causal_attention,
        mask_pad_tokens_in_attention=args.mask_pad_attention,
        mask_intra_frame_pad_tokens_in_attention=args.mask_intra_frame_pad_attention,
        attention_backend=args.attention_backend,
        rope_3d=args.rope_3d,
        rope_base=args.rope_base,
    ).to(device)

    load_base_weights(
        donor_model,
        checkpoint_path=args.base_checkpoint,
        state_key=args.base_state_key,
        strip_prefix=args.base_checkpoint_strip_prefix,
        load_scope=args.base_load_scope,
        is_main=is_main,
    )

    path = build_discrete_path(
        scheduler_name=args.discrete_scheduler,
        discrete_rate=args.discrete_rate,
        discrete_poly_n=args.discrete_poly_n,
        discrete_vp_beta_min=args.discrete_vp_beta_min,
        discrete_vp_beta_max=args.discrete_vp_beta_max,
    )
    loss_function = get_loss_function(args.loss_function, path=path)

    selected_layers, importance_scores = choose_selected_layers(
        model=donor_model,
        args=args,
        train_loader=train_loader,
        path=path,
        loss_function=loss_function,
        device=device,
        bf16_autocast=bf16_autocast,
        distributed=distributed,
        rank=rank,
        is_main=is_main,
    )

    model = ARCFlowViTSelectedLayersAdapter(
        selected_layers=selected_layers,
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_colors=args.num_colors,
        max_frames=max_frames,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        framewise_causal_attention=args.framewise_causal_attention,
        mask_pad_tokens_in_attention=args.mask_pad_attention,
        mask_intra_frame_pad_tokens_in_attention=args.mask_intra_frame_pad_attention,
        attention_backend=args.attention_backend,
        rope_3d=args.rope_3d,
        rope_base=args.rope_base,
    ).to(device)

    donor_state = donor_model.state_dict()
    model.load_state_dict(donor_state, strict=False)
    del donor_model

    if args.freeze_selected_base_layers:
        freeze_selected_layers(model, selected_layers)

    lora_targets = _split_csv(args.lora_targets)
    apply_lora_to_selected_layers(
        model,
        selected_layers=selected_layers,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_paths=lora_targets,
        is_main=is_main,
    )

    model.prime_flex_attention_block_masks(frames=max_frames)
    model = maybe_compile_model(model, args, is_main=is_main)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Check freezing/LoRA settings.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    if args.lr_scheduler == "cosine":
        total_train_steps = max(args.epochs * len(train_loader), 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_train_steps,
            eta_min=args.min_learning_rate,
        )

    if is_main:
        trainable_count, total_count = count_trainable_parameters(model)
        print(
            f"Selected layers: {selected_layers} | "
            f"trainable_params={trainable_count:,}/{total_count:,} ({100.0 * trainable_count / max(total_count, 1):.2f}%)"
        )
        importance_path = Path(args.importance_save_path)
        importance_path.parent.mkdir(parents=True, exist_ok=True)
        with importance_path.open("w") as f:
            json.dump(
                {
                    "selected_layers": selected_layers,
                    "strategy": args.layer_selection_strategy,
                    "scores": importance_scores,
                    "max_base_layers": int(args.max_base_layers),
                    "importance_batches": int(args.importance_batches),
                    "base_checkpoint": args.base_checkpoint,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        print(f"Saved layer importance report to {importance_path}")

    wandb_run = None
    if args.use_wandb and is_main:
        if wandb is None:
            raise RuntimeError("wandb is not installed. Install it or disable --use-wandb.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                **vars(args),
                "selected_layers": selected_layers,
            },
        )

    best_eval_loss = float("inf")
    global_step = 0
    eval_round = 0
    eval_on_steps = args.eval_every_steps > 0

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        seen = 0

        train_iter = tqdm(
            train_loader,
            desc=f"train {epoch}/{args.epochs}",
            total=len(train_loader),
            leave=False,
            disable=not is_main,
        )

        for batch_idx, batch in enumerate(train_iter, 1):
            frames = batch["frames"].to(device)
            frame_valid_mask = batch["frame_valid_mask"].to(device)
            target_frame_index = batch["target_frame_index"].to(device)
            target_valid_mask = batch["target_valid_mask"].to(device)

            batch_size, frame_count, _, _ = frames.shape
            frame_times = sample_frame_times(
                batch_size=batch_size,
                frames=frame_count,
                device=device,
                discretization_steps=args.train_time_discretization_steps,
            )
            x_t_tokens = sample_xt_from_qt(
                frames,
                frame_valid_mask,
                frame_times,
                num_colors=args.num_colors,
                path=path,
            )
            x_t = one_hot_frames(x_t_tokens, num_colors=args.num_colors)

            cudagraph_step_begin_if_available()
            with autocast_context(device, bf16_autocast):
                logits = model(x_t, frame_times, frame_valid_mask=frame_valid_mask)
            loss = discrete_flow_matching_loss(
                logits.float(),
                frames,
                x_t_tokens,
                frame_valid_mask=frame_valid_mask,
                frame_times=frame_times,
                target_frame_index=target_frame_index,
                target_valid_mask=target_valid_mask,
                loss_function=loss_function,
                target_only=args.loss_on_target_only,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            loss_value = float(loss.item())
            running_loss += loss_value * batch_size
            seen += batch_size
            global_step += 1

            if is_main and hasattr(train_iter, "set_postfix"):
                train_iter.set_postfix(
                    step=global_step,
                    loss=f"{loss_value:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

            if is_main and args.log_every_steps > 0 and global_step % args.log_every_steps == 0:
                elapsed = time.time() - epoch_start
                print(
                    " | ".join(
                        [
                            f"epoch={epoch}",
                            f"step={global_step}",
                            f"batch={batch_idx}/{len(train_loader)}",
                            f"loss={loss_value:.6f}",
                            f"lr={optimizer.param_groups[0]['lr']:.2e}",
                            f"elapsed={elapsed:.1f}s",
                        ]
                    )
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/step_loss": loss_value,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

            if eval_on_steps and global_step % args.eval_every_steps == 0:
                eval_round += 1
                if eval_sampler is not None:
                    eval_sampler.set_epoch(eval_round)
                eval_loss, eval_metrics = run_evaluation(
                    model=model,
                    eval_loader=eval_loader,
                    fallback_loader=train_loader,
                    device=device,
                    args=args,
                    path=path,
                    loss_function=loss_function,
                    bf16_autocast=bf16_autocast,
                    show_progress=is_main,
                )
                if is_main:
                    if np.isfinite(eval_loss) and eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_metrics = dict(eval_metrics)
                        best_metrics["eval_loss"] = eval_loss
                        save_checkpoint(
                            save_path=Path(args.best_save_path),
                            model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            args=args,
                            metrics=best_metrics,
                        )
                    print(
                        " | ".join(
                            [
                                "eval(trigger=steps)",
                                f"epoch={epoch}",
                                f"step={global_step}",
                                f"loss={eval_loss:.6f}",
                                f"sample_acc={eval_metrics['sample_acc']:.4f}",
                                f"task_acc={eval_metrics['task_acc']:.4f}",
                            ]
                        )
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "eval/loss": eval_loss,
                                "eval/sample_acc": eval_metrics["sample_acc"],
                                "eval/task_acc": eval_metrics["task_acc"],
                                "eval/trigger_step": global_step,
                                "eval/trigger_epoch": epoch,
                            },
                            step=global_step,
                        )
                model.train()

        if distributed:
            totals = torch.tensor([running_loss, float(seen)], device=device)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            running_loss = float(totals[0].item())
            seen = int(totals[1].item())

        train_loss = running_loss / max(seen, 1)
        epoch_time = time.time() - epoch_start
        log_data: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "epoch_time": epoch_time,
            "lr": optimizer.param_groups[0]["lr"],
        }

        should_eval = (not eval_on_steps) and args.eval_every > 0 and (epoch % args.eval_every == 0)
        if should_eval:
            eval_round += 1
            if eval_sampler is not None:
                eval_sampler.set_epoch(eval_round)
            eval_loss, eval_metrics = run_evaluation(
                model=model,
                eval_loader=eval_loader,
                fallback_loader=train_loader,
                device=device,
                args=args,
                path=path,
                loss_function=loss_function,
                bf16_autocast=bf16_autocast,
                show_progress=is_main,
            )
            if is_main:
                log_data.update(
                    {
                        "eval_loss": eval_loss,
                        "eval_sample_acc": eval_metrics["sample_acc"],
                        "eval_task_acc": eval_metrics["task_acc"],
                    }
                )
                if np.isfinite(eval_loss) and eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_metrics = dict(eval_metrics)
                    best_metrics["eval_loss"] = eval_loss
                    save_checkpoint(
                        save_path=Path(args.best_save_path),
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        args=args,
                        metrics=best_metrics,
                    )

        if is_main:
            if args.verbose:
                print(
                    " | ".join(
                        [
                            f"epoch={log_data['epoch']}",
                            f"loss={log_data['train_loss']:.6f}",
                            f"time={log_data['epoch_time']:.1f}s",
                            f"lr={log_data['lr']:.6f}",
                            f"eval_loss={log_data.get('eval_loss', float('nan')):.6f}",
                            f"sample_acc={log_data.get('eval_sample_acc', float('nan')):.4f}",
                            f"task_acc={log_data.get('eval_task_acc', float('nan')):.4f}",
                        ]
                    )
                )

            if wandb_run is not None:
                wandb_payload = dict(log_data)
                if "eval_loss" in log_data:
                    wandb_payload["eval/loss"] = log_data["eval_loss"]
                    wandb_payload["eval/sample_acc"] = log_data.get("eval_sample_acc", float("nan"))
                    wandb_payload["eval/task_acc"] = log_data.get("eval_task_acc", float("nan"))
                wandb_run.log(wandb_payload, step=global_step)

            save_checkpoint(
                save_path=Path(args.save_path),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                metrics=log_data,
            )

    if wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed(distributed)


if __name__ == "__main__":
    train(parse_args())
