import argparse
import copy
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.distributions import Categorical

from src.ARC_loader import IGNORE_INDEX, PAD_INDEX, build_dataloaders
from utils.args import build_parser
from utils.distribution import init_distributed_mode
from utils.load_model import load_model_only
from utils.vlm_reward import QwenYesNoRewardModel, TaskContextCache, compose_task_image, render_arc_grid

try:
    import wandb
except ImportError:
    wandb = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_rl_args(parser: argparse.ArgumentParser) -> None:
    parser.description = "RL stage for VARC with GRPO and VLM yes/no reward"
    parser.add_argument("--reward-model-id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--reward-dtype", type=str, default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--reward-use-image", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reward-max-context", type=int, default=4)
    parser.add_argument("--reference-checkpoint", type=str, default=None)
    parser.add_argument("--rl-epochs", type=int, default=1)
    parser.add_argument("--rl-max-steps", type=int, default=0, help="0 means no explicit step cap.")
    parser.add_argument("--rl-group-size", type=int, default=4)
    parser.add_argument("--rl-temperature", type=float, default=1.0)
    parser.add_argument("--rl-clip-eps", type=float, default=0.2)
    parser.add_argument("--rl-beta", type=float, default=0.01, help="KL penalty coefficient.")
    parser.add_argument("--rl-entropy-coef", type=float, default=1e-3)
    parser.add_argument("--rl-update-epochs", type=int, default=2)
    parser.add_argument(
        "--rl-action-mask",
        type=str,
        default="target",
        choices=("full", "target"),
        help="Use 'target' to sample only supervised output region; 'full' samples entire canvas.",
    )
    parser.add_argument("--rl-log-every", type=int, default=1)
    parser.add_argument("--rl-save-every", type=int, default=50)
    parser.add_argument("--rl-save-path", type=str, default="saves/rl_stage/checkpoint_final.pt")
    parser.add_argument("--rl-vis-every", type=int, default=25, help="Log sample visualizations every N RL steps.")
    parser.add_argument("--rl-vis-samples", type=int, default=4, help="Number of sampled rollouts to visualize per logging step.")
    parser.add_argument("--rl-vis-max-context", type=int, default=3, help="Number of demo pairs to include in each visualization.")
    parser.add_argument("--rl-vis-dir", type=str, default="outputs/rl_vis", help="Optional directory for saving rollout visualizations.")
    parser.add_argument(
        "--rl-reward-sanity-every",
        type=int,
        default=25,
        help="Compute sampled-vs-ground-truth reward sanity checks every N steps (0 to disable).",
    )
    parser.add_argument(
        "--rl-reward-sanity-samples",
        type=int,
        default=2,
        help="How many rollouts to evaluate against ground-truth reward at sanity steps.",
    )
    parser.add_argument(
        "--rl-disable-aug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable translation/scale augmentation for RL rollouts.",
    )
    parser.add_argument("--rl-grad-clip", type=float, default=1.0)


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    add_rl_args(parser)
    args = parser.parse_args()
    if not args.resume_checkpoint:
        raise ValueError("--resume-checkpoint is required for RL stage.")
    if args.rl_group_size < 2:
        raise ValueError("--rl-group-size must be at least 2 for GRPO.")
    return args


def _repeat_tensor(x: torch.Tensor, repeats: int) -> torch.Tensor:
    return x.repeat_interleave(repeats, dim=0)


def _repeat_list(values: Sequence[Any], repeats: int) -> List[Any]:
    return [item for item in values for _ in range(repeats)]


def downsample_by_majority(grid: List[List[int]], factor: int) -> List[List[int]]:
    if factor <= 1:
        return grid
    np_grid = np.asarray(grid, dtype=np.int64)
    if np_grid.size == 0:
        return grid
    out: List[List[int]] = []
    for y in range(0, np_grid.shape[0], factor):
        row: List[int] = []
        for x in range(0, np_grid.shape[1], factor):
            block = np_grid[y : y + factor, x : x + factor].reshape(-1)
            if block.size == 0:
                continue
            counts = np.bincount(block)
            row.append(int(np.argmax(counts)))
        if row:
            out.append(row)
    return out if out else [[0]]


def decode_prediction_grid(
    canvas_prediction: np.ndarray,
    offset_xy: Sequence[int],
    scale_factor: int,
) -> List[List[int]]:
    offset_x = int(offset_xy[0])
    offset_y = int(offset_xy[1])
    cropped = canvas_prediction[offset_y:, offset_x:]
    if cropped.size == 0:
        return [[0]]

    width = 0
    while width < cropped.shape[1] and int(cropped[0, width]) != PAD_INDEX:
        width += 1
    height = 0
    while height < cropped.shape[0] and int(cropped[height, 0]) != PAD_INDEX:
        height += 1

    if width == 0:
        width = min(1, cropped.shape[1])
    if height == 0:
        height = min(1, cropped.shape[0])

    prediction = cropped[:height, :width].tolist()
    return downsample_by_majority(prediction, max(1, int(scale_factor)))


@dataclass
class RolloutVisualization:
    image: Image.Image
    task_name: str
    reward: float
    yes_logit: float
    no_logit: float
    example_index: int
    gt_reward: Optional[float] = None
    gt_yes_logit: Optional[float] = None
    gt_no_logit: Optional[float] = None


def build_rollout_visualization(
    *,
    train_examples: Sequence[Dict[str, Any]],
    query_input: Sequence[Sequence[int]],
    candidate_output: Sequence[Sequence[int]],
    ground_truth: Optional[Sequence[Sequence[int]]],
    max_context: int,
) -> Image.Image:
    base_image = compose_task_image(
        train_examples=train_examples,
        query_input=query_input,
        candidate_output=candidate_output,
        max_context=max_context,
    )
    if ground_truth is None:
        return base_image

    gt_image = render_arc_grid(ground_truth)
    panel_height = gt_image.height + 2 * 14
    panel = Image.new("RGB", (base_image.width, panel_height), color=(255, 255, 255))
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.text((12, 8), "ground truth output", fill=(0, 0, 0))
    panel.paste(gt_image, (12, 24))

    merged = Image.new("RGB", (base_image.width, base_image.height + panel.height + 8), color=(255, 255, 255))
    merged.paste(base_image, (0, 0))
    merged.paste(panel, (0, base_image.height + 8))
    return merged


def build_action_mask(
    *,
    targets: torch.Tensor,
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    if mode == "full":
        return torch.ones_like(targets, dtype=torch.float32, device=device)
    mask = (targets != IGNORE_INDEX).float()
    empty = mask.flatten(1).sum(dim=1) == 0
    if empty.any():
        mask[empty] = 1.0
    return mask


def gather_action_logprob(
    logits: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    scaled_logits = logits / max(temperature, 1e-6)
    log_probs = F.log_softmax(scaled_logits, dim=1)
    selected = torch.gather(log_probs, dim=1, index=actions.unsqueeze(1)).squeeze(1)
    normalizer = action_mask.sum(dim=(1, 2)).clamp_min(1.0)
    return (selected * action_mask).sum(dim=(1, 2)) / normalizer


def sample_actions(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scaled_logits = logits / max(temperature, 1e-6)
    dist = Categorical(logits=scaled_logits.permute(0, 2, 3, 1))
    actions = dist.sample()
    action_logprob_map = dist.log_prob(actions)
    normalizer = action_mask.sum(dim=(1, 2)).clamp_min(1.0)
    old_logprob = (action_logprob_map * action_mask).sum(dim=(1, 2)) / normalizer
    return actions, old_logprob


def compute_entropy(logits: torch.Tensor, action_mask: torch.Tensor, *, temperature: float) -> torch.Tensor:
    scaled_logits = logits / max(temperature, 1e-6)
    dist = Categorical(logits=scaled_logits.permute(0, 2, 3, 1))
    entropy_map = dist.entropy()
    normalizer = action_mask.sum(dim=(1, 2)).clamp_min(1.0)
    return (entropy_map * action_mask).sum(dim=(1, 2)) / normalizer


@dataclass
class RolloutBatch:
    inputs: torch.Tensor
    attention_mask: torch.Tensor
    task_ids: torch.Tensor
    actions: torch.Tensor
    action_mask: torch.Tensor
    old_logprob: torch.Tensor
    ref_logprob: torch.Tensor
    advantages: torch.Tensor
    rewards: torch.Tensor
    mean_reward: float
    std_reward: float
    mean_yes_logit: float
    mean_no_logit: float
    gt_reward_mean: Optional[float] = None
    reward_sanity_gap: Optional[float] = None
    visualizations: List[RolloutVisualization] = field(default_factory=list)


def collect_rollout(
    *,
    model: torch.nn.Module,
    reference_model: torch.nn.Module,
    reward_model: QwenYesNoRewardModel,
    context_cache: TaskContextCache,
    batch: Dict[str, Any],
    group_size: int,
    action_mask_mode: str,
    temperature: float,
    device: torch.device,
    collect_visualizations: bool = False,
    max_visualizations: int = 0,
    visualization_context: int = 3,
    reward_sanity_samples: int = 0,
) -> RolloutBatch:
    inputs = batch["inputs"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    task_ids = batch["task_ids"].to(device)
    targets = batch["targets"].to(device)
    offsets = batch["offset"].to(device)
    scale_factors = batch["scale_factors"].to(device)
    task_names = batch["task_names"]
    raw_inputs = batch["raw_inputs"]
    raw_outputs = batch["raw_outputs"]

    repeated_inputs = _repeat_tensor(inputs, group_size)
    repeated_mask = _repeat_tensor(attention_mask, group_size)
    repeated_task_ids = _repeat_tensor(task_ids, group_size)
    repeated_targets = _repeat_tensor(targets, group_size)
    repeated_offsets = _repeat_tensor(offsets, group_size)
    repeated_scales = _repeat_tensor(scale_factors, group_size)
    repeated_task_names = _repeat_list(task_names, group_size)
    repeated_raw_inputs = _repeat_list(raw_inputs, group_size)
    repeated_raw_outputs = _repeat_list(raw_outputs, group_size)

    action_mask = build_action_mask(
        targets=repeated_targets,
        mode=action_mask_mode,
        device=device,
    )

    with torch.no_grad():
        logits = model(repeated_inputs, repeated_task_ids, attention_mask=repeated_mask)
        actions, old_logprob = sample_actions(logits, action_mask, temperature=temperature)

        ref_logits = reference_model(repeated_inputs, repeated_task_ids, attention_mask=repeated_mask)
        ref_logprob = gather_action_logprob(ref_logits, actions, action_mask, temperature=temperature)

    rewards: List[float] = []
    yes_logits: List[float] = []
    no_logits: List[float] = []
    visualizations: List[RolloutVisualization] = []

    actions_np = actions.detach().cpu().numpy()
    repeated_offsets_cpu = repeated_offsets.detach().cpu().tolist()
    repeated_scales_cpu = repeated_scales.detach().cpu().tolist()
    for i in range(actions_np.shape[0]):
        task_name = repeated_task_names[i]
        train_examples = context_cache.get_train_examples(task_name)
        query_input = repeated_raw_inputs[i]
        candidate_output = decode_prediction_grid(
            actions_np[i],
            offset_xy=repeated_offsets_cpu[i],
            scale_factor=int(repeated_scales_cpu[i]),
        )
        reward = reward_model.score_candidate(
            train_examples=train_examples,
            query_input=query_input,
            candidate_output=candidate_output,
        )
        rewards.append(reward.reward)
        yes_logits.append(reward.yes_logit)
        no_logits.append(reward.no_logit)
        if collect_visualizations and len(visualizations) < max_visualizations:
            gt_reward_value: Optional[float] = None
            gt_yes: Optional[float] = None
            gt_no: Optional[float] = None
            if repeated_raw_outputs[i] is not None and len(visualizations) < reward_sanity_samples:
                gt_reward = reward_model.score_candidate(
                    train_examples=train_examples,
                    query_input=query_input,
                    candidate_output=repeated_raw_outputs[i],
                )
                gt_reward_value = float(gt_reward.reward)
                gt_yes = float(gt_reward.yes_logit)
                gt_no = float(gt_reward.no_logit)
            viz_image = build_rollout_visualization(
                train_examples=train_examples,
                query_input=query_input,
                candidate_output=candidate_output,
                ground_truth=repeated_raw_outputs[i],
                max_context=visualization_context,
            )
            visualizations.append(
                RolloutVisualization(
                    image=viz_image,
                    task_name=str(task_name),
                    reward=float(reward.reward),
                    yes_logit=float(reward.yes_logit),
                    no_logit=float(reward.no_logit),
                    example_index=i,
                    gt_reward=gt_reward_value,
                    gt_yes_logit=gt_yes,
                    gt_no_logit=gt_no,
                )
            )

    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    grouped = reward_tensor.view(inputs.shape[0], group_size)
    grouped_mean = grouped.mean(dim=1, keepdim=True)
    grouped_std = grouped.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    advantages = ((grouped - grouped_mean) / grouped_std).reshape(-1)

    gt_rewards = [item.gt_reward for item in visualizations if item.gt_reward is not None]
    sampled_rewards_for_gt = [item.reward for item in visualizations if item.gt_reward is not None]
    gt_reward_mean: Optional[float] = None
    reward_sanity_gap: Optional[float] = None
    if gt_rewards and sampled_rewards_for_gt:
        gt_reward_mean = float(np.mean(gt_rewards))
        sampled_mean = float(np.mean(sampled_rewards_for_gt))
        reward_sanity_gap = gt_reward_mean - sampled_mean

    return RolloutBatch(
        inputs=repeated_inputs,
        attention_mask=repeated_mask,
        task_ids=repeated_task_ids,
        actions=actions,
        action_mask=action_mask,
        old_logprob=old_logprob,
        ref_logprob=ref_logprob,
        advantages=advantages,
        rewards=reward_tensor,
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_yes_logit=float(np.mean(yes_logits)),
        mean_no_logit=float(np.mean(no_logits)),
        gt_reward_mean=gt_reward_mean,
        reward_sanity_gap=reward_sanity_gap,
        visualizations=visualizations,
    )


def run_grpo_update(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    update_epochs: int,
    clip_eps: float,
    beta: float,
    entropy_coef: float,
    temperature: float,
    grad_clip: float,
) -> Dict[str, float]:
    advantages = rollout.advantages.detach()
    old_logprob = rollout.old_logprob.detach()
    ref_logprob = rollout.ref_logprob.detach()

    last_policy_loss = 0.0
    last_kl = 0.0
    last_entropy = 0.0
    for _ in range(update_epochs):
        logits = model(rollout.inputs, rollout.task_ids, attention_mask=rollout.attention_mask)
        new_logprob = gather_action_logprob(
            logits,
            rollout.actions,
            rollout.action_mask,
            temperature=temperature,
        )
        ratio = torch.exp(new_logprob - old_logprob)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()

        kl_term = ((new_logprob - ref_logprob) ** 2).mean()
        entropy_term = compute_entropy(logits, rollout.action_mask, temperature=temperature).mean()
        loss = policy_loss + beta * kl_term - entropy_coef * entropy_term

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        last_policy_loss = float(policy_loss.detach().cpu())
        last_kl = float(kl_term.detach().cpu())
        last_entropy = float(entropy_term.detach().cpu())

    return {
        "policy_loss": last_policy_loss,
        "kl": last_kl,
        "entropy": last_entropy,
    }


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    *,
    save_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "args": vars(args),
    }
    torch.save(payload, save_path)


def save_visualizations_locally(
    *,
    visualizations: Sequence[RolloutVisualization],
    output_dir: Path,
    step: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, item in enumerate(visualizations):
        safe_task_name = item.task_name.replace("/", "_")
        file_name = f"step_{step:06d}_{idx:02d}_{safe_task_name}.png"
        item.image.save(output_dir / file_name)


def log_to_wandb(
    *,
    wandb_run: Any,
    step: int,
    epoch: int,
    elapsed: float,
    rollout: RolloutBatch,
    stats: Dict[str, float],
) -> None:
    metrics: Dict[str, Any] = {
        "rl/epoch": epoch,
        "rl/step": step,
        "rl/reward_mean": rollout.mean_reward,
        "rl/reward_std": rollout.std_reward,
        "rl/yes_logit_mean": rollout.mean_yes_logit,
        "rl/no_logit_mean": rollout.mean_no_logit,
        "rl/policy_loss": stats["policy_loss"],
        "rl/kl": stats["kl"],
        "rl/entropy": stats["entropy"],
        "rl/elapsed_sec": elapsed,
    }
    if rollout.gt_reward_mean is not None:
        metrics["rl/reward_gt_mean"] = rollout.gt_reward_mean
    if rollout.reward_sanity_gap is not None:
        metrics["rl/reward_gt_minus_sampled"] = rollout.reward_sanity_gap
    if rollout.visualizations:
        wandb_images = []
        for item in rollout.visualizations:
            caption = f"task={item.task_name} idx={item.example_index} reward={item.reward:.3f} yes={item.yes_logit:.3f} no={item.no_logit:.3f}"
            if item.gt_reward is not None:
                caption += (
                    f" | gt_reward={item.gt_reward:.3f}"
                    f" gt_yes={item.gt_yes_logit:.3f}"
                    f" gt_no={item.gt_no_logit:.3f}"
                )
            wandb_images.append(wandb.Image(item.image, caption=caption))
        metrics["rl/samples"] = wandb_images
    wandb_run.log(metrics, step=step)


def train(args: argparse.Namespace) -> None:
    distributed, rank, world_size, local_rank, device = init_distributed_mode(args)
    if distributed:
        raise RuntimeError("rl_train_ARC.py currently supports single-process training only.")
    if rank != 0:
        return
    set_seed(args.seed)

    train_dataset, train_loader, eval_dataset, eval_loader, _, _ = build_dataloaders(
        args,
        distributed=False,
        rank=rank,
        world_size=world_size,
    )
    if eval_loader is None:
        raise RuntimeError("Evaluation split is required for RL stage. Provide --eval-split.")
    if not Path(args.resume_checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.resume_checkpoint}")
    if args.reference_checkpoint and not Path(args.reference_checkpoint).exists():
        raise FileNotFoundError(f"Reference checkpoint not found: {args.reference_checkpoint}")
    if args.rl_disable_aug:
        train_dataset.disable_translation()
        train_dataset.disable_resolution_augmentation(fix_scale_factor=1)
        if eval_dataset is not None:
            eval_dataset.disable_translation()
            eval_dataset.disable_resolution_augmentation(fix_scale_factor=1)

    args.no_compile = True
    policy_model = load_model_only(
        args=args,
        train_dataset=train_dataset,
        device=device,
        distributed=False,
        rank=rank,
        local_rank=local_rank,
    )
    policy_model.train()

    ref_args = copy.deepcopy(args)
    if args.reference_checkpoint:
        ref_args.resume_checkpoint = args.reference_checkpoint
    reference_model = load_model_only(
        args=ref_args,
        train_dataset=train_dataset,
        device=device,
        distributed=False,
        rank=rank,
        local_rank=local_rank,
    )
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    wandb_run = None
    if args.use_wandb:
        if wandb is None:
            raise RuntimeError("Weights & Biases is not installed. Install wandb or disable --use-wandb.")
        wandb_kwargs: Dict[str, Any] = {
            "project": args.wandb_project,
            "config": dict(vars(args)),
        }
        if args.wandb_run_name:
            wandb_kwargs["name"] = args.wandb_run_name
        wandb_run = wandb.init(**wandb_kwargs)

    context_split = args.eval_split if args.eval_split else args.train_split
    context_cache = TaskContextCache(Path(args.data_root), split=context_split)
    reward_model = QwenYesNoRewardModel(
        model_id=args.reward_model_id,
        device=device,
        dtype=args.reward_dtype,
        use_image=args.reward_use_image,
        max_context_examples=args.reward_max_context,
    )

    max_steps = args.rl_max_steps if args.rl_max_steps > 0 else None
    global_step = 0
    train_start = time.time()
    vis_dir = Path(args.rl_vis_dir) if args.rl_vis_dir else None
    for epoch in range(args.rl_epochs):
        for batch in eval_loader:
            next_step = global_step + 1
            should_visualize = args.rl_vis_every > 0 and next_step % args.rl_vis_every == 0 and args.rl_vis_samples > 0
            should_sanity = (
                args.rl_reward_sanity_every > 0
                and next_step % args.rl_reward_sanity_every == 0
                and args.rl_reward_sanity_samples > 0
            )
            collect_visualizations = should_visualize or should_sanity
            max_visualizations = max(
                args.rl_vis_samples if should_visualize else 0,
                args.rl_reward_sanity_samples if should_sanity else 0,
            )
            reward_sanity_samples = args.rl_reward_sanity_samples if should_sanity else 0
            policy_model.eval()
            rollout = collect_rollout(
                model=policy_model,
                reference_model=reference_model,
                reward_model=reward_model,
                context_cache=context_cache,
                batch=batch,
                group_size=args.rl_group_size,
                action_mask_mode=args.rl_action_mask,
                temperature=args.rl_temperature,
                device=device,
                collect_visualizations=collect_visualizations,
                max_visualizations=max_visualizations,
                visualization_context=args.rl_vis_max_context,
                reward_sanity_samples=reward_sanity_samples,
            )
            policy_model.train()
            stats = run_grpo_update(
                model=policy_model,
                optimizer=optimizer,
                rollout=rollout,
                update_epochs=args.rl_update_epochs,
                clip_eps=args.rl_clip_eps,
                beta=args.rl_beta,
                entropy_coef=args.rl_entropy_coef,
                temperature=args.rl_temperature,
                grad_clip=args.rl_grad_clip,
            )
            global_step += 1
            elapsed = time.time() - train_start

            if vis_dir is not None and rollout.visualizations:
                save_visualizations_locally(
                    visualizations=rollout.visualizations,
                    output_dir=vis_dir,
                    step=global_step,
                )

            if args.rl_log_every > 0 and global_step % args.rl_log_every == 0:
                print(
                    " | ".join(
                        [
                            f"epoch={epoch}",
                            f"step={global_step}",
                            f"reward={rollout.mean_reward:.4f}",
                            f"reward_std={rollout.std_reward:.4f}",
                            f"yes_logit={rollout.mean_yes_logit:.4f}",
                            f"no_logit={rollout.mean_no_logit:.4f}",
                            f"policy_loss={stats['policy_loss']:.4f}",
                            f"kl={stats['kl']:.4f}",
                            f"entropy={stats['entropy']:.4f}",
                            f"gt_gap={rollout.reward_sanity_gap:.4f}" if rollout.reward_sanity_gap is not None else "gt_gap=na",
                            f"elapsed={elapsed:.1f}s",
                        ]
                    )
                )
            if wandb_run is not None:
                log_to_wandb(
                    wandb_run=wandb_run,
                    step=global_step,
                    epoch=epoch,
                    elapsed=elapsed,
                    rollout=rollout,
                    stats=stats,
                )

            if args.rl_save_every > 0 and global_step % args.rl_save_every == 0:
                base_path = Path(args.rl_save_path)
                step_path = base_path.with_name(f"{base_path.stem}_step_{global_step}{base_path.suffix}")
                save_checkpoint(
                    save_path=step_path,
                    model=policy_model,
                    optimizer=optimizer,
                    step=global_step,
                    args=args,
                )

            if max_steps is not None and global_step >= max_steps:
                break
        if max_steps is not None and global_step >= max_steps:
            break

    save_checkpoint(
        save_path=Path(args.rl_save_path),
        model=policy_model,
        optimizer=optimizer,
        step=global_step,
        args=args,
    )
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Saved RL checkpoint to {args.rl_save_path}")


if __name__ == "__main__":
    train(parse_args())
