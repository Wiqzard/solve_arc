from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.ARC_FlowViT import ARCFlowViT
from src.ARC_context_flow_loader import build_flow_context_dataloaders

try:
    import wandb
except ImportError:
    wandb = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--loss-on-target-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply discrete flow-matching loss only on the final solution frame.",
    )
    parser.add_argument("--min-noise-level", type=float, default=1e-3)
    parser.add_argument("--max-noise-level", type=float, default=0.999)
    parser.add_argument(
        "--discrete-rate",
        type=float,
        default=5.0,
        help="CTMC replacement rate beta for q_t = exp(t*R).",
    )
    parser.add_argument(
        "--weight-by-inverse-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Weight CE by 1/(1-sigma_t) to emphasize low-noise denoising consistency.",
    )
    parser.add_argument(
        "--reverse-sampler",
        type=str,
        default="sample",
        choices=("sample", "argmax"),
        help="How to draw x_s from model reverse kernel during evaluation.",
    )

    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--sample-steps", type=int, default=40, help="Discrete reverse steps for last-frame denoising.")

    parser.add_argument("--save-path", type=str, default="saves/flow_context_vit_discrete/checkpoint_last.pt")
    parser.add_argument("--best-save-path", type=str, default="saves/flow_context_vit_discrete/checkpoint_best.pt")

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="VisionARC")
    parser.add_argument("--wandb-run-name", type=str, default="flow-context-vit-discrete")
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


def sigma_from_time(t: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.exp(-beta * t)


def transition_matrix_uniform(
    *,
    t: float | torch.Tensor,
    num_colors: int,
    beta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Q_t where Q_t[i,j] = p(x_t=j | x_0=i)."""
    t_tensor = torch.tensor(float(t), device=device, dtype=dtype) if not torch.is_tensor(t) else t.to(device=device, dtype=dtype)
    sigma = sigma_from_time(t_tensor, beta=beta)
    eye = torch.eye(num_colors, device=device, dtype=dtype)
    ones = torch.ones((num_colors, num_colors), device=device, dtype=dtype) / float(num_colors)
    return sigma * eye + (1.0 - sigma) * ones


def sample_xt_from_qt(
    clean_tokens: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    frame_times: torch.Tensor,
    *,
    num_colors: int,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample x_t from exact CTMC marginal q_t(x_t|x_0)."""
    x0_onehot = one_hot_frames(clean_tokens, num_colors=num_colors)
    sigma = sigma_from_time(frame_times, beta=beta)[:, :, None, None, None]
    q_t_probs = sigma * x0_onehot + (1.0 - sigma) / float(num_colors)
    flat_probs = q_t_probs.reshape(-1, num_colors)
    sampled = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
    x_t = sampled.reshape_as(clean_tokens)

    changed = (x_t != clean_tokens) & frame_valid_mask.bool()
    return x_t, changed, q_t_probs


def discrete_flow_matching_loss(
    logits: torch.Tensor,
    clean_tokens: torch.Tensor,
    *,
    frame_valid_mask: torch.Tensor,
    changed_mask: torch.Tensor,
    target_frame_index: torch.Tensor,
    target_valid_mask: torch.Tensor,
    frame_times: torch.Tensor,
    beta: float,
    target_only: bool,
    weight_by_inverse_noise: bool,
) -> torch.Tensor:
    # logits: (B, F, H, W, C)
    if target_only:
        batch = logits.size(0)
        batch_idx = torch.arange(batch, device=logits.device)
        pred = logits[batch_idx, target_frame_index]  # (B, H, W, C)
        target = clean_tokens[batch_idx, target_frame_index]  # (B, H, W)
        valid = target_valid_mask.bool()
        changed = changed_mask[batch_idx, target_frame_index]

        used = valid & changed
        if not used.any():
            used = valid
        ce = F.cross_entropy(pred.permute(0, 3, 1, 2), target, reduction="none")
        if weight_by_inverse_noise:
            sigma = sigma_from_time(frame_times[batch_idx, target_frame_index], beta=beta).clamp(max=1 - 1e-4)
            weight = (1.0 / (1.0 - sigma))[:, None, None]
        else:
            weight = torch.ones((batch, 1, 1), device=logits.device)
        masked = ce * used.float() * weight
        denom = (used.float() * weight).sum().clamp_min(1.0)
        return masked.sum() / denom

    ce = F.cross_entropy(
        logits.permute(0, 1, 4, 2, 3).reshape(-1, logits.size(-1), logits.size(2), logits.size(3)),
        clean_tokens.reshape(-1, clean_tokens.size(2), clean_tokens.size(3)),
        reduction="none",
    ).reshape_as(clean_tokens).float()
    used = frame_valid_mask.bool() & changed_mask.bool()
    if not used.any():
        used = frame_valid_mask.bool()

    if weight_by_inverse_noise:
        sigma = sigma_from_time(frame_times, beta=beta).clamp(max=1 - 1e-4)
        weight = (1.0 / (1.0 - sigma))[:, :, None, None]
    else:
        weight = torch.ones_like(frame_times)[:, :, None, None]
    masked = ce * used.float() * weight
    denom = (used.float() * weight).sum().clamp_min(1.0)
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

    # Start from q_{t=1} on target frame; keep all other frames clean context.
    random_target = torch.randint(
        low=0,
        high=num_colors,
        size=(batch_size, state.size(2), state.size(3)),
        device=device,
        dtype=state.dtype,
    )
    state[batch_idx, target_frame_index] = random_target

    clean_context = frames.clone()
    target_valid_mask = frame_valid_mask[batch_idx, target_frame_index].bool()
    for step in range(steps, 0, -1):
        t = float(step) / float(steps)
        t_next = float(step - 1) / float(steps)
        frame_times = torch.zeros((batch_size, frame_count), dtype=torch.float32, device=device)
        frame_times[batch_idx, target_frame_index] = t

        state_onehot = one_hot_frames(state, num_colors=num_colors)
        logits = model(state_onehot, frame_times, frame_valid_mask=frame_valid_mask)
        target_logits = logits[batch_idx, target_frame_index]  # (B, H, W, C)
        pred_x0 = torch.softmax(target_logits, dim=-1).reshape(-1, num_colors)  # p_theta(x0 | x_t, t)
        xt_token = state[batch_idx, target_frame_index].reshape(-1)

        # Exact reverse kernel p_theta(x_s | x_t) = sum_{x0} q_{s|t}(x_s|x_t,x0) p_theta(x0|x_t)
        q_s = transition_matrix_uniform(
            t=t_next,
            num_colors=num_colors,
            beta=beta,
            device=device,
            dtype=pred_x0.dtype,
        )
        q_t = transition_matrix_uniform(
            t=t,
            num_colors=num_colors,
            beta=beta,
            device=device,
            dtype=pred_x0.dtype,
        )
        q_delta = transition_matrix_uniform(
            t=max(t - t_next, 0.0),
            num_colors=num_colors,
            beta=beta,
            device=device,
            dtype=pred_x0.dtype,
        )

        # denominator per candidate clean state i: q_t(k|i)
        denom = q_t[:, xt_token].T.clamp_min(1e-8)  # (N, K)
        weighted = pred_x0 / denom  # (N, K)
        mix = weighted @ q_s  # (N, K) over x_s=j
        q_delta_col = q_delta[:, xt_token].T  # (N, K), q_delta(j->k)
        reverse_probs = mix * q_delta_col
        reverse_probs = reverse_probs / reverse_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        if reverse_sampler == "argmax":
            next_token_flat = torch.argmax(reverse_probs, dim=-1)
        else:
            next_token_flat = torch.multinomial(reverse_probs, num_samples=1).squeeze(-1)
        next_token = next_token_flat.reshape(batch_size, state.size(2), state.size(3))

        current = state[batch_idx, target_frame_index]
        current[target_valid_mask] = next_token[target_valid_mask]
        state[batch_idx, target_frame_index] = current

        # Always keep context frames clean.
        for b in range(batch_size):
            ti = int(target_frame_index[b].item())
            if ti > 0:
                state[b, :ti] = clean_context[b, :ti]
            if ti + 1 < frame_count:
                state[b, ti + 1 :] = clean_context[b, ti + 1 :]

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
) -> Dict[str, float]:
    if loader is None:
        return {"sample_acc": 0.0, "task_acc": 0.0, "samples": 0.0}

    task_total: Dict[str, int] = {}
    task_correct: Dict[str, int] = {}
    sample_total = 0
    sample_correct = 0

    for batch in loader:
        frames = batch["frames"].to(device)
        frame_valid_mask = batch["frame_valid_mask"].to(device)
        target_frame_index = batch["target_frame_index"].to(device)
        target_output = batch["target_output"].to(device)
        target_valid_mask = batch["target_valid_mask"].to(device)
        task_names = batch["task_names"]

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

    sample_acc = sample_correct / max(sample_total, 1)
    task_acc = 0.0
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
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "args": vars(args),
        "metrics": metrics or {},
    }
    torch.save(payload, save_path)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    train_dataset, train_loader, eval_dataset, eval_loader = build_flow_context_dataloaders(args)
    max_frames = 2 * args.num_demos + 2
    context_length = 2 * args.num_demos * (args.image_size * args.image_size)
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
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    wandb_run = None
    if args.use_wandb:
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

            x_t_tokens, changed_mask, _ = sample_xt_from_qt(
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
                frame_valid_mask=frame_valid_mask,
                changed_mask=changed_mask,
                target_frame_index=target_frame_index,
                target_valid_mask=target_valid_mask,
                frame_times=frame_times,
                beta=args.discrete_rate,
                target_only=args.loss_on_target_only,
                weight_by_inverse_noise=args.weight_by_inverse_noise,
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

            if args.log_every_steps > 0 and global_step % args.log_every_steps == 0:
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

        train_loss = running_loss / max(seen, 1)
        epoch_time = time.time() - epoch_start
        log_data: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "epoch_time": epoch_time,
            "lr": optimizer.param_groups[0]["lr"],
        }

        if args.eval_every > 0 and (epoch % args.eval_every == 0):
            eval_metrics = evaluate_last_frame_accuracy(
                model,
                eval_loader if eval_loader is not None else train_loader,
                device=device,
                num_colors=args.num_colors,
                sample_steps=args.sample_steps,
                beta=args.discrete_rate,
                reverse_sampler=args.reverse_sampler,
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


if __name__ == "__main__":
    train(parse_args())
