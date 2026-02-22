from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from src.ARC_FlowViT import ARCFlowViT, ARCFlowViTLooped
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


def setup_distributed(args: argparse.Namespace) -> Tuple[bool, int, int, int, torch.device]:
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


def parse_optional_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ARC frame-context ViT from scratch with flow matching.")
    parser.add_argument("--data-root", type=str, default="raw_data/ARC-AGI")
    parser.add_argument("--train-split", type=str, default="training")
    parser.add_argument("--eval-split", type=str, default="evaluation")

    parser.add_argument(
        "--max-demos",
        "--num-demos",
        dest="max_demos",
        type=int,
        default=3,
        help="Maximum number of demonstration pairs (m) in the context.",
    )
    parser.add_argument("--image-size", type=int, default=30)
    parser.add_argument("--num-colors", type=int, default=12)

    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument(
        "--model-arch",
        type=str,
        default="flow_vit",
        choices=("flow_vit", "flow_vit_looped"),
        help="Model architecture variant.",
    )
    parser.add_argument(
        "--n-loops",
        type=int,
        default=2,
        help="Number of repeated passes through the full layer stack (used by flow_vit_looped).",
    )
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--framewise-causal-attention",
        nargs="?",
        const=True,
        default=False,
        type=parse_optional_bool,
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
        "--mask-pad-attention",
        action="store_true",
        default=False,
        help="Mask padded tokens in attention. For flex-causal attention, this is combined with framewise causal masking.",
    )
    parser.add_argument(
        "--rope-3d",
        nargs="?",
        const=True,
        default=False,
        type=parse_optional_bool,
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
        help="Maximum total BARC train queries to include (-1 means all).",
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
        "--nested-dropout",
        default=False,
        nargs="?",
        const=True,
        type=parse_optional_bool,
        help=(
            "Train-time demo dropout: sample k in [1, max_demos], keep at most k demos, "
            "and left-pad remaining demo slots."
        ),
    )

    parser.add_argument(
        "--loss-on-target-only",
        action="store_true",
        default=False,
        help="Apply flow-matching loss only on the final solution frame.",
    )

    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=0,
        help="If > 0, run evaluation every N optimizer steps (disables epoch-based eval).",
    )
    parser.add_argument("--sample-steps", type=int, default=40, help="Euler steps for last-frame denoising.")

    parser.add_argument("--save-path", type=str, default="saves/flow_context_vit/checkpoint_last.pt")
    parser.add_argument("--best-save-path", type=str, default="saves/flow_context_vit/checkpoint_best.pt")

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="VisionARC")
    parser.add_argument("--wandb-run-name", type=str, default="flow_context_vit")
    args = parser.parse_args()
    args.num_demos = args.max_demos
    return args


def one_hot_frames(frames: torch.Tensor, num_colors: int) -> torch.Tensor:
    return F.one_hot(frames.long(), num_classes=num_colors).float()


def sample_frame_times(
    *,
    batch_size: int,
    frames: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.rand(batch_size, frames, device=device)


def build_noisy_state(
    x0: torch.Tensor,
    frame_times: torch.Tensor,
    *,
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if noise is None:
        noise = torch.randn_like(x0)
    t = frame_times[:, :, None, None, None]
    x_t = (1.0 - t) * x0 + t * noise
    target_velocity = noise - x0
    return x_t, target_velocity


def flow_matching_loss(
    pred_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    *,
    frame_valid_mask: torch.Tensor,
    target_frame_index: torch.Tensor,
    target_valid_mask: torch.Tensor,
    target_only: bool,
) -> torch.Tensor:
    if target_only:
        batch = pred_velocity.size(0)
        batch_idx = torch.arange(batch, device=pred_velocity.device)
        pred = pred_velocity[batch_idx, target_frame_index]
        target = target_velocity[batch_idx, target_frame_index]
        mask = target_valid_mask.unsqueeze(-1).float()
        loss = ((pred - target) ** 2 * mask).sum()
        denom = mask.sum().clamp_min(1.0) * pred.size(-1)
        return loss / denom

    mask = frame_valid_mask.unsqueeze(-1).float()
    loss = ((pred_velocity - target_velocity) ** 2 * mask).sum()
    denom = mask.sum().clamp_min(1.0) * pred_velocity.size(-1)
    return loss / denom


@torch.no_grad()
def denoise_last_solution_frame(
    model: ARCFlowViT,
    *,
    frames: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    target_frame_index: torch.Tensor,
    num_colors: int,
    steps: int,
    autocast_enabled: bool = False,
) -> torch.Tensor:
    model.eval()
    x0 = one_hot_frames(frames, num_colors=num_colors)
    state = x0.clone()
    batch_size, frame_count, _, _, _ = state.shape
    device = state.device
    batch_idx = torch.arange(batch_size, device=device)

    state[batch_idx, target_frame_index] = torch.randn_like(state[batch_idx, target_frame_index])

    dt = 1.0 / float(steps)
    for step in range(steps, 0, -1):
        t = float(step) / float(steps)
        frame_times = torch.zeros((batch_size, frame_count), dtype=torch.float32, device=device)
        frame_times[batch_idx, target_frame_index] = t

        cudagraph_step_begin_if_available()
        with autocast_context(device, autocast_enabled):
            pred_velocity = model(state, frame_times, frame_valid_mask=frame_valid_mask)
        state[batch_idx, target_frame_index] = state[batch_idx, target_frame_index] - pred_velocity[batch_idx, target_frame_index] * dt

        for b in range(batch_size):
            ti = int(target_frame_index[b].item())
            if ti > 0:
                state[b, :ti] = x0[b, :ti]
            if ti + 1 < frame_count:
                state[b, ti + 1 :] = x0[b, ti + 1 :]

    predicted_last = state[batch_idx, target_frame_index].argmax(dim=-1)
    return predicted_last


@torch.no_grad()
def evaluate_last_frame_accuracy(
    model: ARCFlowViT,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    num_colors: int,
    sample_steps: int,
    show_progress: bool = True,
    autocast_enabled: bool = False,
) -> Dict[str, float]:
    if loader is None:
        return {
            "sample_acc": 0.0,
            "sample_acc_at_50": 0.0,
            "sample_acc_at_80": 0.0,
            "sample_acc_at_90": 0.0,
            "sample_acc_at_95": 0.0,
            "task_acc": 0.0,
            "samples": 0.0,
        }

    episode_results: Dict[str, tuple[str, bool, bool, bool, bool, bool]] = {}

    eval_iterator = tqdm(loader, desc="eval", total=len(loader), leave=False, disable=not show_progress)
    for batch in eval_iterator:
        frames = batch["frames"].to(device)
        frame_valid_mask = batch["frame_valid_mask"].to(device)
        target_frame_index = batch["target_frame_index"].to(device)
        target_output = batch["target_output"].to(device)
        target_valid_mask = batch["target_valid_mask"].to(device)
        task_names = batch["task_names"]
        query_indices = batch["query_indices"]

        prediction = denoise_last_solution_frame(
            model,
            frames=frames,
            frame_valid_mask=frame_valid_mask,
            target_frame_index=target_frame_index,
            num_colors=num_colors,
            steps=sample_steps,
            autocast_enabled=autocast_enabled,
        )

        valid = target_valid_mask.bool()
        exact = (((prediction == target_output) | ~valid).view(prediction.size(0), -1)).all(dim=1)
        match_ratio = (
            ((prediction == target_output) & valid).view(prediction.size(0), -1).float().sum(dim=1)
            / valid.view(prediction.size(0), -1).float().sum(dim=1).clamp_min(1.0)
        )
        at_50 = match_ratio >= 0.50
        at_80 = match_ratio >= 0.80
        at_90 = match_ratio >= 0.90
        at_95 = match_ratio >= 0.95

        for i in range(prediction.size(0)):
            task_name = task_names[i]
            is_correct = bool(exact[i].item())
            is_at_50 = bool(at_50[i].item())
            is_at_80 = bool(at_80[i].item())
            is_at_90 = bool(at_90[i].item())
            is_at_95 = bool(at_95[i].item())
            query_index = int(query_indices[i].item())
            episode_key = f"{task_name}::{query_index}"
            episode_results[episode_key] = (task_name, is_correct, is_at_50, is_at_80, is_at_90, is_at_95)

    if dist.is_available() and dist.is_initialized():
        gathered: list[Optional[Dict[str, tuple[str, bool, bool, bool, bool, bool]]]] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, episode_results)
        merged_results: Dict[str, tuple[str, bool, bool, bool, bool, bool]] = {}
        for shard in gathered:
            if shard is not None:
                merged_results.update(shard)
    else:
        merged_results = episode_results

    sample_total = len(merged_results)
    sample_correct = sum(1 for _, is_correct, _, _, _, _ in merged_results.values() if is_correct)
    sample_correct_at_50 = sum(1 for _, _, is_at_50, _, _, _ in merged_results.values() if is_at_50)
    sample_correct_at_80 = sum(1 for _, _, _, is_at_80, _, _ in merged_results.values() if is_at_80)
    sample_correct_at_90 = sum(1 for _, _, _, _, is_at_90, _ in merged_results.values() if is_at_90)
    sample_correct_at_95 = sum(1 for _, _, _, _, _, is_at_95 in merged_results.values() if is_at_95)
    sample_acc = sample_correct / max(sample_total, 1)
    sample_acc_at_50 = sample_correct_at_50 / max(sample_total, 1)
    sample_acc_at_80 = sample_correct_at_80 / max(sample_total, 1)
    sample_acc_at_90 = sample_correct_at_90 / max(sample_total, 1)
    sample_acc_at_95 = sample_correct_at_95 / max(sample_total, 1)
    task_acc = 0.0
    task_total: Dict[str, int] = {}
    task_correct: Dict[str, int] = {}
    for task_name, is_correct, _, _, _, _ in merged_results.values():
        task_total[task_name] = task_total.get(task_name, 0) + 1
        task_correct[task_name] = task_correct.get(task_name, 0) + int(is_correct)
    if task_total:
        task_acc = float(np.mean([task_correct[name] / task_total[name] for name in task_total]))
    return {
        "sample_acc": sample_acc,
        "sample_acc_at_50": sample_acc_at_50,
        "sample_acc_at_80": sample_acc_at_80,
        "sample_acc_at_90": sample_acc_at_90,
        "sample_acc_at_95": sample_acc_at_95,
        "task_acc": task_acc,
        "samples": float(sample_total),
    }


@torch.no_grad()
def evaluate_flow_matching_loss(
    model: ARCFlowViT,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    num_colors: int,
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

        x0 = one_hot_frames(frames, num_colors=num_colors)
        batch_size, frame_count, _, _, _ = x0.shape
        frame_times = sample_frame_times(
            batch_size=batch_size,
            frames=frame_count,
            device=device,
        )
        x_t, target_velocity = build_noisy_state(x0, frame_times)
        cudagraph_step_begin_if_available()
        with autocast_context(device, autocast_enabled):
            pred_velocity = model(x_t, frame_times, frame_valid_mask=frame_valid_mask)
        loss = flow_matching_loss(
            pred_velocity.float(),
            target_velocity,
            frame_valid_mask=frame_valid_mask,
            target_frame_index=target_frame_index,
            target_valid_mask=target_valid_mask,
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
    max_frames = 2 * args.num_demos + 2  # demos (2m frames) + query input + query output
    context_length = 2 * args.num_demos * (args.image_size * args.image_size)
    if is_main:
        print(f"Flow context tokens (demo-only): {context_length}")
        print(f"Full sequence tokens with query pair: {(max_frames) * (args.image_size * args.image_size)}")
        print(f"Train episodes: {len(train_dataset)}")
        if eval_dataset is not None:
            print(f"Eval episodes: {len(eval_dataset)}")

    model_cls = ARCFlowViTLooped if args.model_arch == "flow_vit_looped" else ARCFlowViT
    model_loops = args.n_loops if args.model_arch == "flow_vit_looped" else 1
    if is_main:
        print(f"Model architecture: {args.model_arch} (n_loops={model_loops})")
    model = model_cls(
        image_size=args.image_size,
        num_colors=args.num_colors,
        max_frames=max_frames,
        embed_dim=args.embed_dim,
        depth=args.depth,
        n_loops=model_loops,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        framewise_causal_attention=args.framewise_causal_attention,
        mask_pad_tokens_in_attention=args.mask_pad_attention,
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
    if is_main and eval_on_steps:
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
        for batch_idx, batch in enumerate(train_loader, 1):
            frames = batch["frames"].to(device)
            frame_valid_mask = batch["frame_valid_mask"].to(device)
            target_frame_index = batch["target_frame_index"].to(device)
            target_valid_mask = batch["target_valid_mask"].to(device)

            x0 = one_hot_frames(frames, num_colors=args.num_colors)
            batch_size, frame_count, _, _, _ = x0.shape
            frame_times = sample_frame_times(
                batch_size=batch_size,
                frames=frame_count,
                device=device,
            )

            x_t, target_velocity = build_noisy_state(x0, frame_times)
            cudagraph_step_begin_if_available()
            with autocast_context(device, bf16_autocast):
                pred_velocity = model(x_t, frame_times, frame_valid_mask=frame_valid_mask)
            loss = flow_matching_loss(
                pred_velocity.float(),
                target_velocity,
                frame_valid_mask=frame_valid_mask,
                target_frame_index=target_frame_index,
                target_valid_mask=target_valid_mask,
                target_only=args.loss_on_target_only,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            batch_size = frames.size(0)
            loss_value = float(loss.item())
            running_loss += loss_value * batch_size
            seen += batch_size
            global_step += 1
            step_loss_accum += loss_value
            step_loss_count += 1

            if is_main and args.log_every_steps > 0 and global_step % args.log_every_steps == 0:
                step_avg_loss = step_loss_accum / max(step_loss_count, 1)
                elapsed = time.time() - epoch_start
                print(
                    " | ".join(
                        [
                            f"epoch={epoch}",
                            f"step={global_step}",
                            f"batch={batch_idx}/{total_batches}",
                            f"step_loss={loss_value:.6f}",
                            f"step_avg_loss={step_avg_loss:.6f}",
                            f"elapsed={elapsed:.1f}s",
                        ]
                    )
                )
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

            if eval_on_steps and global_step % args.eval_every_steps == 0:
                # Free large per-step training tensors before eval to avoid transient OOM spikes.
                del frames, frame_valid_mask, target_frame_index, target_valid_mask
                del x0, frame_times, x_t, target_velocity, pred_velocity, loss
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                eval_round += 1
                if eval_sampler is not None:
                    eval_sampler.set_epoch(eval_round)
                eval_loss = evaluate_flow_matching_loss(
                    model,
                    eval_loader if eval_loader is not None else train_loader,
                    device=device,
                    num_colors=args.num_colors,
                    target_only=args.loss_on_target_only,
                    show_progress=False,
                    autocast_enabled=bf16_autocast,
                )
                eval_metrics = evaluate_last_frame_accuracy(
                    model,
                    eval_loader if eval_loader is not None else train_loader,
                    device=device,
                    num_colors=args.num_colors,
                    sample_steps=args.sample_steps,
                    show_progress=is_main,
                    autocast_enabled=bf16_autocast,
                )
                if is_main:
                    print(
                        " | ".join(
                            [
                                "eval(trigger=steps)",
                                f"epoch={epoch}",
                                f"step={global_step}",
                                f"loss={eval_loss:.6f}",
                                f"sample_acc={eval_metrics['sample_acc']:.4f}",
                                f"sample_acc@50={eval_metrics['sample_acc_at_50']:.4f}",
                                f"sample_acc@80={eval_metrics['sample_acc_at_80']:.4f}",
                                f"sample_acc@90={eval_metrics['sample_acc_at_90']:.4f}",
                                f"sample_acc@95={eval_metrics['sample_acc_at_95']:.4f}",
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
                                "eval/sample_acc_at_50": eval_metrics["sample_acc_at_50"],
                                "eval/sample_acc_at_80": eval_metrics["sample_acc_at_80"],
                                "eval/sample_acc_at_90": eval_metrics["sample_acc_at_90"],
                                "eval/sample_acc_at_95": eval_metrics["sample_acc_at_95"],
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
            eval_loss = evaluate_flow_matching_loss(
                model,
                eval_loader if eval_loader is not None else train_loader,
                device=device,
                num_colors=args.num_colors,
                target_only=args.loss_on_target_only,
                show_progress=False,
                autocast_enabled=bf16_autocast,
            )
            eval_metrics = evaluate_last_frame_accuracy(
                model,
                eval_loader if eval_loader is not None else train_loader,
                device=device,
                num_colors=args.num_colors,
                sample_steps=args.sample_steps,
                show_progress=is_main,
                autocast_enabled=bf16_autocast,
            )
            if is_main:
                log_data.update(
                    {
                        "eval_loss": eval_loss,
                        "eval_sample_acc": eval_metrics["sample_acc"],
                        "eval_sample_acc_at_50": eval_metrics["sample_acc_at_50"],
                        "eval_sample_acc_at_80": eval_metrics["sample_acc_at_80"],
                        "eval_sample_acc_at_90": eval_metrics["sample_acc_at_90"],
                        "eval_sample_acc_at_95": eval_metrics["sample_acc_at_95"],
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
            print(
                " | ".join(
                    [
                        f"epoch={log_data['epoch']}",
                        f"loss={log_data['train_loss']:.6f}",
                        f"time={log_data['epoch_time']:.1f}s",
                        f"lr={log_data['lr']:.6f}",
                        f"eval_loss={log_data.get('eval_loss', float('nan')):.6f}",
                        f"sample_acc={log_data.get('eval_sample_acc', float('nan')):.4f}",
                        f"sample_acc@50={log_data.get('eval_sample_acc_at_50', float('nan')):.4f}",
                        f"sample_acc@80={log_data.get('eval_sample_acc_at_80', float('nan')):.4f}",
                        f"sample_acc@90={log_data.get('eval_sample_acc_at_90', float('nan')):.4f}",
                        f"sample_acc@95={log_data.get('eval_sample_acc_at_95', float('nan')):.4f}",
                        f"task_acc={log_data.get('eval_task_acc', float('nan')):.4f}",
                    ]
                )
            )

            if wandb_run is not None:
                wandb_payload = dict(log_data)
                if "eval_loss" in log_data:
                    wandb_payload["eval/loss"] = log_data["eval_loss"]
                    wandb_payload["eval/sample_acc"] = log_data.get("eval_sample_acc", float("nan"))
                    wandb_payload["eval/sample_acc_at_50"] = log_data.get("eval_sample_acc_at_50", float("nan"))
                    wandb_payload["eval/sample_acc_at_80"] = log_data.get("eval_sample_acc_at_80", float("nan"))
                    wandb_payload["eval/sample_acc_at_90"] = log_data.get("eval_sample_acc_at_90", float("nan"))
                    wandb_payload["eval/sample_acc_at_95"] = log_data.get("eval_sample_acc_at_95", float("nan"))
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
