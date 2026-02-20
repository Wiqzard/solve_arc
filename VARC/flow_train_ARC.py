from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ARC frame-context ViT from scratch with flow matching.")
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
        help="Apply flow-matching loss only on the final solution frame.",
    )
    parser.add_argument("--min-noise-level", type=float, default=1e-3)
    parser.add_argument("--max-noise-level", type=float, default=0.999)

    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--sample-steps", type=int, default=40, help="Euler steps for last-frame denoising.")

    parser.add_argument("--save-path", type=str, default="saves/flow_context_vit/checkpoint_last.pt")
    parser.add_argument("--best-save-path", type=str, default="saves/flow_context_vit/checkpoint_best.pt")

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="VisionARC")
    parser.add_argument("--wandb-run-name", type=str, default="flow_context_vit")
    return parser.parse_args()


def one_hot_frames(frames: torch.Tensor, num_colors: int) -> torch.Tensor:
    return F.one_hot(frames.long(), num_classes=num_colors).float()


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
) -> Dict[str, float]:
    if loader is None:
        return {"sample_acc": 0.0, "task_acc": 0.0, "samples": 0.0}

    episode_results: Dict[str, tuple[str, bool]] = {}

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
        )

        valid = target_valid_mask.bool()
        exact = (((prediction == target_output) | ~valid).view(prediction.size(0), -1)).all(dim=1)

        for i in range(prediction.size(0)):
            task_name = task_names[i]
            is_correct = bool(exact[i].item())
            query_index = int(query_indices[i].item())
            episode_key = f"{task_name}::{query_index}"
            episode_results[episode_key] = (task_name, is_correct)

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
    return {"sample_acc": sample_acc, "task_acc": task_acc, "samples": float(sample_total)}


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

            x0 = one_hot_frames(frames, num_colors=args.num_colors)
            batch_size, frame_count, _, _, _ = x0.shape
            frame_times = sample_frame_times(
                batch_size=batch_size,
                frames=frame_count,
                device=device,
                min_t=args.min_noise_level,
                max_t=args.max_noise_level,
            )

            x_t, target_velocity = build_noisy_state(x0, frame_times)
            pred_velocity = model(x_t, frame_times, frame_valid_mask=frame_valid_mask)
            loss = flow_matching_loss(
                pred_velocity,
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
        if should_eval:
            if eval_sampler is not None:
                eval_sampler.set_epoch(epoch)
            eval_metrics = evaluate_last_frame_accuracy(
                model,
                eval_loader if eval_loader is not None else train_loader,
                device=device,
                num_colors=args.num_colors,
                sample_steps=args.sample_steps,
                show_progress=is_main,
            )
            if is_main:
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
