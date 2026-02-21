from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed(args: argparse.Namespace) -> tuple[bool, int, int, int, torch.device]:
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = bool(args.ddp or env_world_size > 1)
    if not distributed:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 0, 1, device

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("DDP requires torchrun environment variables RANK and WORLD_SIZE.")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])
    backend = args.dist_backend
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    dist.init_process_group(backend=backend, init_method=args.dist_url)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return True, rank, local_rank, world_size, device


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def cudagraph_step_begin_if_available() -> None:
    compiler = getattr(torch, "compiler", None)
    if compiler is None:
        return
    mark_step_begin = getattr(compiler, "cudagraph_mark_step_begin", None)
    if callable(mark_step_begin):
        mark_step_begin()


def maybe_compile_model(
    model: torch.nn.Module,
    args: argparse.Namespace,
    *,
    is_main: bool,
) -> torch.nn.Module:
    if not args.compile:
        return model
    if not hasattr(torch, "compile"):
        if is_main:
            print("Warning: torch.compile is unavailable in this PyTorch build; continuing without compile.")
        return model
    try:
        if is_main:
            print(f"Applying torch.compile(mode={args.compile_mode})...")
        return torch.compile(model, mode=args.compile_mode)
    except Exception as exc:
        if is_main:
            print(f"Warning: torch.compile failed ({exc}); continuing without compile.")
        return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ARC frame-context ViT with discrete flow matching.")
    parser.add_argument("--data-root", type=str, default="raw_data/ARC-AGI")
    parser.add_argument("--train-split", type=str, default="training")
    parser.add_argument("--eval-split", type=str, default="evaluation")

    parser.add_argument("--num-demos", type=int, default=3, help="m demonstration pairs.")
    parser.add_argument("--image-size", type=int, default=30)
    parser.add_argument("--num-colors", type=int, default=12)

    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--framewise-causal-attention",
        action="store_true",
        default=False,
        help="Enable framewise causal attention (frame f attends only to frames <= f).",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="auto",
        choices=("auto", "flex", "sdpa"),
        help="Attention backend for framewise-causal mode.",
    )
    parser.add_argument(
        "--rope-3d",
        action="store_true",
        help="Enable 3D RoPE (frame,y,x) on attention q/k.",
    )
    parser.add_argument(
        "--rope-base",
        type=float,
        default=10000.0,
        help="Base frequency for 3D RoPE.",
    )

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable detailed training prints in addition to tqdm bars.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=("cosine", "none"))
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile optimization.")
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=("default", "reduce-overhead", "max-autotune"),
        help="torch.compile mode.",
    )
    parser.add_argument(
        "--include-rearc",
        action="store_true",
        help="Add tasks from the RE-ARC dataset to the flow training set.",
    )
    parser.add_argument(
        "--rearc-path",
        type=str,
        default="raw_data/re_arc",
        help="Path to RE-ARC dataset root.",
    )
    parser.add_argument(
        "--rearc-limit",
        type=int,
        default=-1,
        help="Maximum RE-ARC examples per task (-1 means all).",
    )
    parser.add_argument(
        "--include-barc",
        action="store_true",
        help="Add tasks from the BARC dataset to the flow training set.",
    )
    parser.add_argument(
        "--barc-path",
        type=str,
        default="raw_data/BARC",
        help="Path to BARC dataset root.",
    )
    parser.add_argument(
        "--barc-limit",
        type=int,
        default=-1,
        help="Maximum BARC examples per task (-1 means all).",
    )
    parser.add_argument("--ddp", action="store_true", help="Enable DDP training (torchrun).")
    parser.add_argument("--dist-backend", type=str, default="nccl", choices=("nccl", "gloo"))
    parser.add_argument("--dist-url", type=str, default="env://")
    parser.add_argument("--bf16-autocast", action="store_true", help="Enable bfloat16 autocast on CUDA.")
    parser.add_argument(
        "--flow-train-translation-aug",
        action="store_true",
        default=False,
        help="Enable random translation augmentation for flow train episodes.",
    )
    parser.add_argument(
        "--flow-train-resolution-aug",
        action="store_true",
        default=False,
        help="Enable random resolution scaling augmentation for flow train episodes.",
    )

    parser.add_argument(
        "--loss-on-target-only",
        action="store_true",
        default=False,
        help="Apply discrete flow-matching loss only on the final solution frame.",
    )
    parser.add_argument("--min-noise-level", type=float, default=1e-3)
    parser.add_argument("--max-noise-level", type=float, default=0.999)
    parser.add_argument(
        "--discrete-rate",
        type=float,
        default=5.0,
        help="CTMC jump rate beta for kappa_t = 1-exp(-beta*t).",
    )
    parser.add_argument(
        "--reverse-sampler",
        type=str,
        default="sample",
        choices=("sample", "argmax"),
        help="How to choose x_1 from p_theta(x_1|x_t,t) in Euler sampling.",
    )

    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=0,
        help="If > 0, run evaluation every N optimizer steps (disables epoch-based eval).",
    )
    parser.add_argument("--sample-steps", type=int, default=40, help="Discrete Euler steps for last-frame generation.")

    parser.add_argument("--save-path", type=str, default="saves/flow_context_vit_discrete/checkpoint_last.pt")
    parser.add_argument("--best-save-path", type=str, default="saves/flow_context_vit_discrete/checkpoint_best.pt")

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="VisionARC")
    parser.add_argument("--wandb-run-name", type=str, default="flow-context-vit-discrete")
    parser.add_argument(
        "--wandb-num-vis-samples",
        type=int,
        default=8,
        help="Number of eval episodes to visualize and log to W&B each eval.",
    )
    parser.add_argument(
        "--wandb-vis-scale",
        type=int,
        default=8,
        help="Pixel upscale factor for logged ARC image panels.",
    )
    parser.add_argument(
        "--wandb-train-vis-every-steps",
        type=int,
        default=0,
        help="If > 0, log training-step visualizations every N optimizer steps.",
    )
    parser.add_argument(
        "--wandb-train-vis-samples",
        type=int,
        default=2,
        help="Number of samples from the current train batch to visualize.",
    )
    return parser.parse_args()


def one_hot_frames(frames: torch.Tensor, num_colors: int) -> torch.Tensor:
    return F.one_hot(frames.long(), num_classes=num_colors).float()


ARC_PALETTE = np.asarray(
    [
        [0, 0, 0],
        [0, 116, 217],
        [255, 65, 54],
        [46, 204, 64],
        [255, 220, 0],
        [170, 170, 170],
        [240, 18, 190],
        [255, 133, 27],
        [127, 219, 255],
        [135, 12, 37],
        [255, 255, 255],
        [111, 111, 111],
    ],
    dtype=np.uint8,
)
INVALID_COLOR = np.asarray([225, 225, 225], dtype=np.uint8)


def panel_order_string(num_demos: int) -> str:
    parts: List[str] = []
    for demo_id in range(1, num_demos + 1):
        parts.extend([f"D{demo_id}-in", f"D{demo_id}-out"])
    parts.extend(["Q-in", "Pred", "GT"])
    return ",".join(parts)


def train_panel_order_string(num_demos: int) -> str:
    parts: List[str] = []
    for demo_id in range(1, num_demos + 1):
        parts.extend([f"D{demo_id}-in", f"D{demo_id}-out"])
    parts.extend(["Q-in", "Q-noisy", "Q-pred", "Q-gt"])
    return ",".join(parts)


def render_grid_rgb(
    grid: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    scale: int,
    num_colors: int,
) -> np.ndarray:
    grid_np = grid.detach().cpu().numpy().astype(np.int64)
    mask_np = valid_mask.detach().cpu().numpy().astype(bool)
    palette = ARC_PALETTE
    if num_colors > palette.shape[0]:
        repeats = (num_colors + palette.shape[0] - 1) // palette.shape[0]
        palette = np.tile(palette, (repeats, 1))
    rgb = palette[np.clip(grid_np, 0, num_colors - 1)]
    rgb[~mask_np] = INVALID_COLOR
    rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    return rgb


def make_image_row(images: List[np.ndarray], gap: int = 4) -> np.ndarray:
    if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    h = max(img.shape[0] for img in images)
    w = sum(img.shape[1] for img in images) + gap * max(len(images) - 1, 0)
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    cursor = 0
    for img in images:
        ih, iw = img.shape[:2]
        canvas[:ih, cursor : cursor + iw] = img
        cursor += iw + gap
    return canvas


def make_image_grid(images: List[np.ndarray], cols: int = 4, gap: int = 4) -> np.ndarray:
    if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    rows: List[np.ndarray] = []
    for start in range(0, len(images), cols):
        rows.append(make_image_row(images[start : start + cols], gap=gap))
    w = max(row.shape[1] for row in rows)
    h = sum(row.shape[0] for row in rows) + gap * max(len(rows) - 1, 0)
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    cursor = 0
    for row in rows:
        rh, rw = row.shape[:2]
        canvas[cursor : cursor + rh, :rw] = row
        cursor += rh + gap
    return canvas


def build_eval_visualization(
    *,
    frames: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    prediction: torch.Tensor,
    target_output: torch.Tensor,
    target_valid_mask: torch.Tensor,
    num_demos: int,
    scale: int,
    num_colors: int,
) -> np.ndarray:
    panels: List[np.ndarray] = []
    for demo_idx in range(num_demos):
        input_idx = 2 * demo_idx
        output_idx = input_idx + 1
        panels.append(
            render_grid_rgb(
                frames[input_idx],
                frame_valid_mask[input_idx],
                scale=scale,
                num_colors=num_colors,
            )
        )
        panels.append(
            render_grid_rgb(
                frames[output_idx],
                frame_valid_mask[output_idx],
                scale=scale,
                num_colors=num_colors,
            )
        )

    query_input_idx = 2 * num_demos
    panels.append(
        render_grid_rgb(
            frames[query_input_idx],
            frame_valid_mask[query_input_idx],
            scale=scale,
            num_colors=num_colors,
        )
    )
    panels.append(
        render_grid_rgb(
            prediction,
            target_valid_mask,
            scale=scale,
            num_colors=num_colors,
        )
    )
    panels.append(
        render_grid_rgb(
            target_output,
            target_valid_mask,
            scale=scale,
            num_colors=num_colors,
        )
    )

    return make_image_grid(panels, cols=4, gap=4)


def build_train_step_visualization(
    *,
    frames: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    noisy_target: torch.Tensor,
    prediction: torch.Tensor,
    target_output: torch.Tensor,
    target_valid_mask: torch.Tensor,
    num_demos: int,
    scale: int,
    num_colors: int,
) -> np.ndarray:
    panels: List[np.ndarray] = []
    for demo_idx in range(num_demos):
        input_idx = 2 * demo_idx
        output_idx = input_idx + 1
        panels.append(
            render_grid_rgb(
                frames[input_idx],
                frame_valid_mask[input_idx],
                scale=scale,
                num_colors=num_colors,
            )
        )
        panels.append(
            render_grid_rgb(
                frames[output_idx],
                frame_valid_mask[output_idx],
                scale=scale,
                num_colors=num_colors,
            )
        )

    query_input_idx = 2 * num_demos
    panels.append(
        render_grid_rgb(
            frames[query_input_idx],
            frame_valid_mask[query_input_idx],
            scale=scale,
            num_colors=num_colors,
        )
    )
    panels.append(
        render_grid_rgb(
            noisy_target,
            target_valid_mask,
            scale=scale,
            num_colors=num_colors,
        )
    )
    panels.append(
        render_grid_rgb(
            prediction,
            target_valid_mask,
            scale=scale,
            num_colors=num_colors,
        )
    )
    panels.append(
        render_grid_rgb(
            target_output,
            target_valid_mask,
            scale=scale,
            num_colors=num_colors,
        )
    )
    return make_image_grid(panels, cols=4, gap=4)


def sample_frame_times(
    *,
    batch_size: int,
    frames: int,
    device: torch.device,
    min_t: float,
    max_t: float,
) -> torch.Tensor:
    times = torch.rand(batch_size, frames, device=device)
    return times * (max_t - min_t) + min_t


def sigma_from_time(t: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.exp(-beta * t)


def sample_xt_from_qt(
    clean_tokens: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    frame_times: torch.Tensor,
    *,
    num_colors: int,
    beta: float,
) -> torch.Tensor:
    """
    Sample x_t from a mixture discrete path:
    x_t = x_0 (uniform source) with prob sigma_t, else x_1 (clean target).
    """
    source_tokens = torch.randint(
        low=0,
        high=num_colors,
        size=clean_tokens.shape,
        device=clean_tokens.device,
        dtype=clean_tokens.dtype,
    )
    sigma = sigma_from_time(frame_times, beta=beta)[:, :, None, None]
    source_mask = torch.rand(clean_tokens.shape, device=clean_tokens.device) < sigma
    x_t = torch.where(source_mask, source_tokens, clean_tokens)

    # Keep padding area unchanged; those positions are always masked out of the loss.
    valid = frame_valid_mask.bool()
    return torch.where(valid, x_t, clean_tokens)


def discrete_flow_matching_loss(
    logits: torch.Tensor,
    clean_tokens: torch.Tensor,
    x_t_tokens: torch.Tensor,
    *,
    frame_valid_mask: torch.Tensor,
    target_frame_index: torch.Tensor,
    target_valid_mask: torch.Tensor,
    beta: float,
    target_only: bool,
) -> torch.Tensor:
    # logits: (B, F, H, W, C), predicts p_theta(x_1 | x_t, t)
    def generalized_kl_per_token(
        *,
        log_probs: torch.Tensor,
        probs: torch.Tensor,
        x1_tokens: torch.Tensor,
        xt_tokens: torch.Tensor,
    ) -> torch.Tensor:
        p1_xt = probs.gather(dim=-1, index=xt_tokens.unsqueeze(-1)).squeeze(-1)
        log_p1_x1 = log_probs.gather(dim=-1, index=x1_tokens.unsqueeze(-1)).squeeze(-1)
        delta = (xt_tokens == x1_tokens).float()
        return -beta * (p1_xt - delta + (1.0 - delta) * log_p1_x1)

    if target_only:
        batch = logits.size(0)
        batch_idx = torch.arange(batch, device=logits.device)
        pred = logits[batch_idx, target_frame_index]
        target = clean_tokens[batch_idx, target_frame_index]
        xt = x_t_tokens[batch_idx, target_frame_index]
        valid = target_valid_mask.bool()
        log_probs = F.log_softmax(pred, dim=-1)
        probs = torch.exp(log_probs)
        loss_map = generalized_kl_per_token(log_probs=log_probs, probs=probs, x1_tokens=target, xt_tokens=xt)
        masked = loss_map * valid.float()
        denom = valid.float().sum().clamp_min(1.0)
        return masked.sum() / denom

    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    loss_map = generalized_kl_per_token(log_probs=log_probs, probs=probs, x1_tokens=clean_tokens, xt_tokens=x_t_tokens)
    valid = frame_valid_mask.bool()
    masked = loss_map * valid.float()
    denom = valid.float().sum().clamp_min(1.0)
    return masked.sum() / denom


@torch.no_grad()
def denoise_last_solution_frame_discrete(
    model: ARCFlowViT,
    *,
    frames: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    target_frame_index: torch.Tensor,
    num_colors: int,
    steps: int,
    beta: float,
    reverse_sampler: str,
    autocast_enabled: bool = False,
) -> torch.Tensor:
    model.eval()
    state = frames.clone()
    batch_size, frame_count, _, _ = state.shape
    device = state.device
    batch_idx = torch.arange(batch_size, device=device)

    # Start from source distribution (uniform random tokens) on the target frame.
    random_target = torch.randint(
        low=0,
        high=num_colors,
        size=(batch_size, state.size(2), state.size(3)),
        device=device,
        dtype=state.dtype,
    )
    state[batch_idx, target_frame_index] = random_target

    target_valid_mask = frame_valid_mask[batch_idx, target_frame_index].bool()
    dt = 1.0 / float(max(steps, 1))
    jump_prob = 1.0 - np.exp(-beta * dt)
    for step in range(steps):
        t = float(step) / float(steps)
        frame_times = torch.zeros((batch_size, frame_count), dtype=torch.float32, device=device)
        frame_times[batch_idx, target_frame_index] = t

        state_onehot = one_hot_frames(state, num_colors=num_colors)
        cudagraph_step_begin_if_available()
        with autocast_context(device, autocast_enabled):
            logits = model(state_onehot, frame_times, frame_valid_mask=frame_valid_mask)
        target_logits = logits[batch_idx, target_frame_index].float()
        pred_x1_probs = torch.softmax(target_logits, dim=-1).reshape(-1, num_colors)
        if reverse_sampler == "argmax":
            proposed_flat = torch.argmax(pred_x1_probs, dim=-1)
        else:
            proposed_flat = torch.multinomial(pred_x1_probs, num_samples=1).squeeze(-1)
        proposed = proposed_flat.reshape(batch_size, state.size(2), state.size(3))

        current = state[batch_idx, target_frame_index]
        if step == steps - 1:
            current[target_valid_mask] = proposed[target_valid_mask]
        else:
            jump_mask = (
                torch.rand(current.shape, device=device) < jump_prob
            ) & target_valid_mask & (proposed != current)
            current[jump_mask] = proposed[jump_mask]
        state[batch_idx, target_frame_index] = current

    return state[batch_idx, target_frame_index]


@torch.no_grad()
def evaluate_last_frame_accuracy(
    model: ARCFlowViT,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    num_colors: int,
    sample_steps: int,
    beta: float,
    reverse_sampler: str,
    collect_examples: int = 0,
    show_progress: bool = True,
    autocast_enabled: bool = False,
) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
    if loader is None:
        return {"sample_acc": 0.0, "task_acc": 0.0, "samples": 0.0}, []

    episode_results: Dict[str, tuple[str, bool]] = {}
    examples: List[Dict[str, Any]] = []

    eval_iterator = tqdm(loader, desc="eval", total=len(loader), leave=False, disable=not show_progress)
    for batch in eval_iterator:
        frames = batch["frames"].to(device)
        frame_valid_mask = batch["frame_valid_mask"].to(device)
        target_frame_index = batch["target_frame_index"].to(device)
        target_output = batch["target_output"].to(device)
        target_valid_mask = batch["target_valid_mask"].to(device)
        task_names = batch["task_names"]
        query_indices = batch["query_indices"]

        prediction = denoise_last_solution_frame_discrete(
            model,
            frames=frames,
            frame_valid_mask=frame_valid_mask,
            target_frame_index=target_frame_index,
            num_colors=num_colors,
            steps=sample_steps,
            beta=beta,
            reverse_sampler=reverse_sampler,
            autocast_enabled=autocast_enabled,
        )

        valid = target_valid_mask.bool()
        exact = (((prediction == target_output) | ~valid).view(prediction.size(0), -1)).all(dim=1)

        for i in range(prediction.size(0)):
            task_name = task_names[i]
            is_correct = bool(exact[i].item())
            query_index = int(query_indices[i].item())
            episode_key = f"{task_name}::{query_index}"
            episode_results[episode_key] = (task_name, is_correct)
            if len(examples) < collect_examples:
                examples.append(
                    {
                        "task_name": task_name,
                        "query_index": int(query_indices[i].item()),
                        "is_correct": is_correct,
                        "frames": frames[i].detach().cpu(),
                        "frame_valid_mask": frame_valid_mask[i].detach().cpu(),
                        "prediction": prediction[i].detach().cpu(),
                        "target_output": target_output[i].detach().cpu(),
                        "target_valid_mask": target_valid_mask[i].detach().cpu(),
                    }
                )

    if dist.is_available() and dist.is_initialized():
        gathered: list[Optional[Dict[str, tuple[str, bool]]]] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, episode_results)
        merged_results: Dict[str, tuple[str, bool]] = {}
        for shard in gathered:
            if shard is not None:
                merged_results.update(shard)
    else:
        merged_results = episode_results

    sample_total = len(merged_results)
    sample_correct = sum(1 for _, is_correct in merged_results.values() if is_correct)
    sample_acc = sample_correct / max(sample_total, 1)
    task_acc = 0.0
    task_total: Dict[str, int] = {}
    task_correct: Dict[str, int] = {}
    for task_name, is_correct in merged_results.values():
        task_total[task_name] = task_total.get(task_name, 0) + 1
        task_correct[task_name] = task_correct.get(task_name, 0) + int(is_correct)
    if task_total:
        task_acc = float(np.mean([task_correct[name] / task_total[name] for name in task_total]))
    return {"sample_acc": sample_acc, "task_acc": task_acc, "samples": float(sample_total)}, examples


@torch.no_grad()
def evaluate_discrete_flow_loss(
    model: ARCFlowViT,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    num_colors: int,
    min_noise_level: float,
    max_noise_level: float,
    beta: float,
    target_only: bool,
    show_progress: bool = False,
    autocast_enabled: bool = False,
) -> float:
    if loader is None:
        return float("nan")

    running_loss = 0.0
    seen = 0
    eval_iterator = tqdm(loader, desc="eval_loss", total=len(loader), leave=False, disable=not show_progress)
    for batch in eval_iterator:
        frames = batch["frames"].to(device)
        frame_valid_mask = batch["frame_valid_mask"].to(device)
        target_frame_index = batch["target_frame_index"].to(device)
        target_valid_mask = batch["target_valid_mask"].to(device)

        batch_size, frame_count, _, _ = frames.shape
        frame_times = sample_frame_times(
            batch_size=batch_size,
            frames=frame_count,
            device=device,
            min_t=min_noise_level,
            max_t=max_noise_level,
        )
        x_t_tokens = sample_xt_from_qt(
            frames,
            frame_valid_mask,
            frame_times,
            num_colors=num_colors,
            beta=beta,
        )
        x_t = one_hot_frames(x_t_tokens, num_colors=num_colors)
        cudagraph_step_begin_if_available()
        with autocast_context(device, autocast_enabled):
            logits = model(x_t, frame_times, frame_valid_mask=frame_valid_mask)
        loss = discrete_flow_matching_loss(
            logits.float(),
            frames,
            x_t_tokens,
            frame_valid_mask=frame_valid_mask,
            target_frame_index=target_frame_index,
            target_valid_mask=target_valid_mask,
            beta=beta,
            target_only=target_only,
        )

        running_loss += float(loss.item()) * batch_size
        seen += batch_size

    if dist.is_available() and dist.is_initialized():
        totals = torch.tensor([running_loss, float(seen)], device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        running_loss = float(totals[0].item())
        seen = int(totals[1].item())
    return running_loss / max(seen, 1)


def save_checkpoint(
    *,
    save_path: Path,
    model: ARCFlowViT,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = unwrap_model(model)
    payload = {
        "model_state": model_to_save.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "args": vars(args),
        "metrics": metrics or {},
    }
    torch.save(payload, save_path)


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
    context_length = 2 * args.num_demos * (args.image_size * args.image_size)
    if is_main and args.verbose:
        print(f"Discrete flow context tokens (demo-only): {context_length}")
        print(f"Full sequence tokens with query pair: {max_frames * (args.image_size * args.image_size)}")
        print(f"Train episodes: {len(train_dataset)}")
        if eval_dataset is not None:
            print(f"Eval episodes: {len(eval_dataset)}")

    model = ARCFlowViT(
        image_size=args.image_size,
        num_colors=args.num_colors,
        max_frames=max_frames,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        framewise_causal_attention=args.framewise_causal_attention,
        attention_backend=args.attention_backend,
        rope_3d=args.rope_3d,
        rope_base=args.rope_base,
    ).to(device)
    model = maybe_compile_model(model, args, is_main=is_main)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
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

    wandb_run = None
    if args.use_wandb and is_main:
        if wandb is None:
            raise RuntimeError("wandb is not installed. Install it or disable --use-wandb.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    best_eval_loss = float("inf")
    global_step = 0
    eval_round = 0
    eval_on_steps = args.eval_every_steps > 0
    if is_main and eval_on_steps and args.verbose:
        print(f"Step-based eval enabled: evaluating every {args.eval_every_steps} steps.")
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        seen = 0
        step_loss_accum = 0.0
        step_loss_count = 0

        total_batches = len(train_loader)
        train_iterator = tqdm(
            train_loader,
            desc=f"train {epoch}/{args.epochs}",
            total=total_batches,
            leave=False,
            disable=not is_main,
        )
        for batch_idx, batch in enumerate(train_iterator, 1):
            frames = batch["frames"].to(device)
            frame_valid_mask = batch["frame_valid_mask"].to(device)
            target_frame_index = batch["target_frame_index"].to(device)
            target_valid_mask = batch["target_valid_mask"].to(device)

            batch_size, frame_count, _, _ = frames.shape
            frame_times = sample_frame_times(
                batch_size=batch_size,
                frames=frame_count,
                device=device,
                min_t=args.min_noise_level,
                max_t=args.max_noise_level,
            )

            x_t_tokens = sample_xt_from_qt(
                frames,
                frame_valid_mask,
                frame_times,
                num_colors=args.num_colors,
                beta=args.discrete_rate,
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
                target_frame_index=target_frame_index,
                target_valid_mask=target_valid_mask,
                beta=args.discrete_rate,
                target_only=args.loss_on_target_only,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            loss_value = float(loss.item())
            running_loss += loss_value * batch_size
            seen += batch_size
            global_step += 1
            step_loss_accum += loss_value
            step_loss_count += 1
            if is_main and hasattr(train_iterator, "set_postfix"):
                train_iterator.set_postfix(
                    step=global_step,
                    loss=f"{loss_value:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

            if is_main and args.log_every_steps > 0 and global_step % args.log_every_steps == 0:
                step_avg_loss = step_loss_accum / max(step_loss_count, 1)
                elapsed = time.time() - epoch_start
                log_line = " | ".join(
                    [
                        f"epoch={epoch}",
                        f"step={global_step}",
                        f"batch={batch_idx}/{total_batches}",
                        f"step_loss={loss_value:.6f}",
                        f"step_avg_loss={step_avg_loss:.6f}",
                        f"elapsed={elapsed:.1f}s",
                    ]
                )
                if args.verbose:
                    if hasattr(train_iterator, "write"):
                        train_iterator.write(log_line)
                    else:
                        print(log_line)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/step_loss": loss_value,
                            "train/step_avg_loss": step_avg_loss,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )
                step_loss_accum = 0.0
                step_loss_count = 0

            if (
                is_main
                and wandb_run is not None
                and args.wandb_train_vis_every_steps > 0
                and global_step % args.wandb_train_vis_every_steps == 0
            ):
                num_vis = min(int(args.wandb_train_vis_samples), frames.size(0))
                if num_vis > 0:
                    train_viz_images = []
                    pred_tokens = logits.float().argmax(dim=-1).detach()
                    for sample_idx in range(num_vis):
                        target_idx = int(target_frame_index[sample_idx].item())
                        image = build_train_step_visualization(
                            frames=frames[sample_idx].detach().cpu(),
                            frame_valid_mask=frame_valid_mask[sample_idx].detach().cpu(),
                            noisy_target=x_t_tokens[sample_idx, target_idx].detach().cpu(),
                            prediction=pred_tokens[sample_idx, target_idx].detach().cpu(),
                            target_output=frames[sample_idx, target_idx].detach().cpu(),
                            target_valid_mask=target_valid_mask[sample_idx].detach().cpu(),
                            num_demos=args.num_demos,
                            scale=max(int(args.wandb_vis_scale), 1),
                            num_colors=args.num_colors,
                        )
                        caption = (
                            f"train_step={global_step} | sample={sample_idx} | "
                            f"order={train_panel_order_string(args.num_demos)}"
                        )
                        train_viz_images.append(wandb.Image(image, caption=caption))
                    wandb_run.log(
                        {
                            "train/step_visualizations": train_viz_images,
                            "train/num_step_visualized": len(train_viz_images),
                        },
                        step=global_step,
                    )

            if eval_on_steps and global_step % args.eval_every_steps == 0:
                eval_round += 1
                if eval_sampler is not None:
                    eval_sampler.set_epoch(eval_round)
                eval_loss = evaluate_discrete_flow_loss(
                    model,
                    eval_loader if eval_loader is not None else train_loader,
                    device=device,
                    num_colors=args.num_colors,
                    min_noise_level=args.min_noise_level,
                    max_noise_level=args.max_noise_level,
                    beta=args.discrete_rate,
                    target_only=args.loss_on_target_only,
                    show_progress=False,
                    autocast_enabled=bf16_autocast,
                )
                eval_metrics, eval_examples = evaluate_last_frame_accuracy(
                    model,
                    eval_loader if eval_loader is not None else train_loader,
                    device=device,
                    num_colors=args.num_colors,
                    sample_steps=args.sample_steps,
                    beta=args.discrete_rate,
                    reverse_sampler=args.reverse_sampler,
                    collect_examples=args.wandb_num_vis_samples if (wandb_run is not None and is_main) else 0,
                    show_progress=is_main,
                    autocast_enabled=bf16_autocast,
                )
                if is_main:
                    if args.verbose:
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
                                "eval_loss": eval_loss,
                                "eval/loss": eval_loss,
                                "eval/sample_acc": eval_metrics["sample_acc"],
                                "eval/task_acc": eval_metrics["task_acc"],
                                "eval/trigger_step": global_step,
                                "eval/trigger_epoch": epoch,
                            },
                            step=global_step,
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
                    if wandb_run is not None and eval_examples:
                        viz_images = []
                        for sample in eval_examples:
                            image = build_eval_visualization(
                                frames=sample["frames"],
                                frame_valid_mask=sample["frame_valid_mask"],
                                prediction=sample["prediction"],
                                target_output=sample["target_output"],
                                target_valid_mask=sample["target_valid_mask"],
                                num_demos=args.num_demos,
                                scale=max(int(args.wandb_vis_scale), 1),
                                num_colors=args.num_colors,
                            )
                            caption = (
                                f"task={sample['task_name']} | query={sample['query_index']} | "
                                f"correct={int(sample['is_correct'])} | order={panel_order_string(args.num_demos)}"
                            )
                            viz_images.append(wandb.Image(image, caption=caption))
                        wandb_run.log({"eval/generations": viz_images}, step=global_step)
                        wandb_run.log({"eval/num_visualized": len(viz_images)}, step=global_step)
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
            eval_loss = evaluate_discrete_flow_loss(
                model,
                eval_loader if eval_loader is not None else train_loader,
                device=device,
                num_colors=args.num_colors,
                min_noise_level=args.min_noise_level,
                max_noise_level=args.max_noise_level,
                beta=args.discrete_rate,
                target_only=args.loss_on_target_only,
                show_progress=False,
                autocast_enabled=bf16_autocast,
            )
            eval_metrics, eval_examples = evaluate_last_frame_accuracy(
                model,
                eval_loader if eval_loader is not None else train_loader,
                device=device,
                num_colors=args.num_colors,
                sample_steps=args.sample_steps,
                beta=args.discrete_rate,
                reverse_sampler=args.reverse_sampler,
                collect_examples=args.wandb_num_vis_samples if (wandb_run is not None and is_main) else 0,
                show_progress=is_main,
                autocast_enabled=bf16_autocast,
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
                if wandb_run is not None and eval_examples:
                    viz_images = []
                    for sample in eval_examples:
                        image = build_eval_visualization(
                            frames=sample["frames"],
                            frame_valid_mask=sample["frame_valid_mask"],
                            prediction=sample["prediction"],
                            target_output=sample["target_output"],
                            target_valid_mask=sample["target_valid_mask"],
                            num_demos=args.num_demos,
                            scale=max(int(args.wandb_vis_scale), 1),
                            num_colors=args.num_colors,
                        )
                        caption = (
                            f"task={sample['task_name']} | query={sample['query_index']} | "
                            f"correct={int(sample['is_correct'])} | order={panel_order_string(args.num_demos)}"
                        )
                        viz_images.append(wandb.Image(image, caption=caption))
                    wandb_run.log({"eval/generations": viz_images}, step=global_step)
                    wandb_run.log({"eval/num_visualized": len(viz_images)}, step=global_step)

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
