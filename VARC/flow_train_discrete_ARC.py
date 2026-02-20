from __future__ import annotations

import argparse
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
        action=argparse.BooleanOptionalAction,
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

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddp", action="store_true", help="Enable DDP training (torchrun).")
    parser.add_argument("--dist-backend", type=str, default="nccl", choices=("nccl", "gloo"))
    parser.add_argument("--dist-url", type=str, default="env://")

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
        logits = model(state_onehot, frame_times, frame_valid_mask=frame_valid_mask)
        target_logits = logits[batch_idx, target_frame_index]
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
) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
    if loader is None:
        return {"sample_acc": 0.0, "task_acc": 0.0, "samples": 0.0}, []

    task_total: Dict[str, int] = {}
    task_correct: Dict[str, int] = {}
    sample_total = 0
    sample_correct = 0
    examples: List[Dict[str, Any]] = []

    eval_iterator = tqdm(loader, desc="eval", total=len(loader), leave=False)
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
        )

        valid = target_valid_mask.bool()
        exact = (((prediction == target_output) | ~valid).view(prediction.size(0), -1)).all(dim=1)

        for i in range(prediction.size(0)):
            task_name = task_names[i]
            is_correct = bool(exact[i].item())
            task_total[task_name] = task_total.get(task_name, 0) + 1
            task_correct[task_name] = task_correct.get(task_name, 0) + int(is_correct)
            sample_total += 1
            sample_correct += int(is_correct)
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

    sample_acc = sample_correct / max(sample_total, 1)
    task_acc = 0.0
    if task_total:
        task_acc = float(np.mean([task_correct[name] / task_total[name] for name in task_total]))
    return {"sample_acc": sample_acc, "task_acc": task_acc, "samples": float(sample_total)}, examples


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

    train_dataset, train_loader, eval_dataset, eval_loader, train_sampler = build_flow_context_dataloaders(
        args,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    max_frames = 2 * args.num_demos + 2
    context_length = 2 * args.num_demos * (args.image_size * args.image_size)
    if is_main:
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
    ).to(device)
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

    wandb_run = None
    if args.use_wandb and is_main:
        if wandb is None:
            raise RuntimeError("wandb is not installed. Install it or disable --use-wandb.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    best_task_acc = float("-inf")
    global_step = 0
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
            logits = model(x_t, frame_times, frame_valid_mask=frame_valid_mask)
            loss = discrete_flow_matching_loss(
                logits,
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

        should_eval = args.eval_every > 0 and (epoch % args.eval_every == 0)
        if should_eval and distributed:
            dist.barrier()
        if should_eval and is_main:
            eval_metrics, eval_examples = evaluate_last_frame_accuracy(
                model,
                eval_loader if eval_loader is not None else train_loader,
                device=device,
                num_colors=args.num_colors,
                sample_steps=args.sample_steps,
                beta=args.discrete_rate,
                reverse_sampler=args.reverse_sampler,
                collect_examples=args.wandb_num_vis_samples if wandb_run is not None else 0,
            )
            log_data.update(
                {
                    "eval_sample_acc": eval_metrics["sample_acc"],
                    "eval_task_acc": eval_metrics["task_acc"],
                }
            )
            if eval_metrics["task_acc"] > best_task_acc:
                best_task_acc = eval_metrics["task_acc"]
                save_checkpoint(
                    save_path=Path(args.best_save_path),
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    args=args,
                    metrics=eval_metrics,
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
        if should_eval and distributed:
            dist.barrier()

        if is_main:
            print(
                " | ".join(
                    [
                        f"epoch={log_data['epoch']}",
                        f"loss={log_data['train_loss']:.6f}",
                        f"time={log_data['epoch_time']:.1f}s",
                        f"lr={log_data['lr']:.6f}",
                        f"sample_acc={log_data.get('eval_sample_acc', float('nan')):.4f}",
                        f"task_acc={log_data.get('eval_task_acc', float('nan')):.4f}",
                    ]
                )
            )

            if wandb_run is not None:
                wandb_run.log(log_data, step=global_step)

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
