from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
import random
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.nn.parallel import DistributedDataParallel as DDP

from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import (
    CondOTScheduler,
    CosineScheduler,
    ExponentialScheduler,
    LinearVPScheduler,
    PolynomialConvexScheduler,
    VPScheduler,
)
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from src.ARC_FlowViT import ARCFlowViT, ARCFlowViTLooped
from src.ARC_context_flow_loader import build_flow_context_dataloaders

# Silence noisy pydantic internals warnings frequently emitted via wandb dependency stack.
warnings.filterwarnings(
    "ignore",
    message=r".*attribute with value .* was provided to the .*Field\(\).* function, which has no effect.*",
    module=r"pydantic\._internal\._generate_schema",
)

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
    parser = argparse.ArgumentParser(description="Train ARC frame-context ViT with discrete flow matching.")
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
        default=256.0,
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
        help="Apply discrete flow-matching loss only on the final solution frame.",
    )
    parser.add_argument(
        "--loss-function",
        type=str,
        default="generalized_kl",
        choices=("cross_entropy", "generalized_kl"),
        help="Discrete flow-matching loss function (Meta-style selector).",
    )
    parser.add_argument(
        "--discrete-rate",
        type=float,
        default=5.0,
        help="Exponential scheduler rate for alpha_t = 1 - exp(-beta*t).",
    )
    parser.add_argument(
        "--discrete-scheduler",
        type=str,
        default="exponential",
        choices=("exponential", "condot", "polynomial", "vp", "linear_vp", "cosine"),
        help="Scheduler used by MixtureDiscreteProbPath during training and sampling.",
    )
    parser.add_argument(
        "--discrete-poly-n",
        type=float,
        default=2.0,
        help="Polynomial degree n for --discrete-scheduler polynomial.",
    )
    parser.add_argument(
        "--discrete-vp-beta-min",
        type=float,
        default=0.1,
        help="beta_min for --discrete-scheduler vp.",
    )
    parser.add_argument(
        "--discrete-vp-beta-max",
        type=float,
        default=20.0,
        help="beta_max for --discrete-scheduler vp.",
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
    parser.add_argument(
        "--train-time-discretization-steps",
        type=int,
        default=1000,
        help="Number of discrete time bins for sampling training/eval-loss times t in [0, 1).",
    )

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
    args = parser.parse_args()
    # Backward-compatibility for existing code paths and scripts still using args.num_demos.
    args.num_demos = args.max_demos
    return args


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
        parts.extend([f"D{demo_id}-in(x_t)", f"D{demo_id}-out(x_t)"])
    parts.extend(["Q-in(x_t)", "Q-target(x_t)", "Q-pred", "Q-gt"])
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
    clean_frames: torch.Tensor,
    noisy_frames: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    target_frame_index: int,
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
                noisy_frames[input_idx],
                frame_valid_mask[input_idx],
                scale=scale,
                num_colors=num_colors,
            )
        )
        panels.append(
            render_grid_rgb(
                noisy_frames[output_idx],
                frame_valid_mask[output_idx],
                scale=scale,
                num_colors=num_colors,
            )
        )

    query_input_idx = 2 * num_demos
    panels.append(
        render_grid_rgb(
            noisy_frames[query_input_idx],
            frame_valid_mask[query_input_idx],
            scale=scale,
            num_colors=num_colors,
        )
    )
    panels.append(
        render_grid_rgb(
            noisy_frames[target_frame_index],
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
            target_output if target_output is not None else clean_frames[target_frame_index],
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
    discretization_steps: int,
) -> torch.Tensor:
    if discretization_steps <= 1:
        return torch.zeros(batch_size, frames, device=device)
    indices = torch.randint(0, discretization_steps, (batch_size, frames), device=device)
    return indices.float() / float(discretization_steps)


def build_discrete_path(
    *,
    scheduler_name: str,
    discrete_rate: float,
    discrete_poly_n: float,
    discrete_vp_beta_min: float,
    discrete_vp_beta_max: float,
) -> MixtureDiscreteProbPath:
    if scheduler_name == "exponential":
        scheduler = ExponentialScheduler(beta=discrete_rate)
    elif scheduler_name == "condot":
        scheduler = CondOTScheduler()
    elif scheduler_name == "polynomial":
        scheduler = PolynomialConvexScheduler(n=discrete_poly_n)
    elif scheduler_name == "vp":
        scheduler = VPScheduler(beta_min=discrete_vp_beta_min, beta_max=discrete_vp_beta_max)
    elif scheduler_name == "linear_vp":
        scheduler = LinearVPScheduler()
    elif scheduler_name == "cosine":
        scheduler = CosineScheduler()
    else:
        raise ValueError(f"Unsupported discrete scheduler: {scheduler_name}")
    return MixtureDiscreteProbPath(scheduler=scheduler)


def get_loss_function(loss_function: str, path: Optional[MixtureDiscreteProbPath] = None) -> _Loss:
    if loss_function == "cross_entropy":
        return torch.nn.CrossEntropyLoss(reduction="none")
    if loss_function == "generalized_kl":
        if path is None:
            raise ValueError("path must be provided for generalized_kl loss")
        return MixturePathGeneralizedKL(path=path, reduction="none")
    raise ValueError(f"{loss_function} is not supported")


def sample_xt_from_qt(
    clean_tokens: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    frame_times: torch.Tensor,
    *,
    num_colors: int,
    path: MixtureDiscreteProbPath,
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
    x_t = path.sample(x_0=source_tokens, x_1=clean_tokens, t=frame_times).x_t

    # Keep padding area unchanged; those positions are always masked out of the loss.
    valid = frame_valid_mask.bool()
    return torch.where(valid, x_t, clean_tokens)


def discrete_flow_matching_loss(
    logits: torch.Tensor,
    clean_tokens: torch.Tensor,
    x_t_tokens: torch.Tensor,
    *,
    frame_valid_mask: torch.Tensor,
    frame_times: torch.Tensor,
    target_frame_index: torch.Tensor,
    target_valid_mask: torch.Tensor,
    loss_function: _Loss,
    target_only: bool,
) -> torch.Tensor:
    # Use Meta-style loss classes on flattened per-frame sequences.
    batch_size, frame_count, height, width, num_colors = logits.shape
    seq_len = height * width
    logits_flat = logits.reshape(batch_size * frame_count, seq_len, num_colors)
    x1_flat = clean_tokens.reshape(batch_size * frame_count, seq_len)
    if isinstance(loss_function, MixturePathGeneralizedKL):
        xt_flat = x_t_tokens.reshape(batch_size * frame_count, seq_len)
        t_flat = frame_times.reshape(batch_size * frame_count)
        loss_map = loss_function(logits_flat, x1_flat, xt_flat, t_flat).reshape(batch_size, frame_count, height, width)
    elif isinstance(loss_function, torch.nn.CrossEntropyLoss):
        ce_map = loss_function(logits_flat.permute(0, 2, 1), x1_flat)
        loss_map = ce_map.reshape(batch_size, frame_count, height, width)
    else:
        raise TypeError(f"Unsupported loss module type: {type(loss_function)}")

    if target_only:
        batch_idx = torch.arange(batch_size, device=logits.device)
        target_loss = loss_map[batch_idx, target_frame_index]
        valid = target_valid_mask.bool()
        masked = target_loss * valid.float()
        denom = valid.float().sum().clamp_min(1.0)
        return masked.sum() / denom

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
    path: MixtureDiscreteProbPath,
    reverse_sampler: str,
    autocast_enabled: bool = False,
) -> torch.Tensor:
    model.eval()
    batch_size, frame_count, height, width = frames.shape
    device = frames.device
    batch_idx = torch.arange(batch_size, device=device)
    target_valid_mask = frame_valid_mask[batch_idx, target_frame_index].bool()
    step_size = 1.0 / float(max(steps, 1))

    class ARCPosteriorWrapper(ModelWrapper):
        def __init__(self, flow_model: ARCFlowViT, vocab_size: int, force_argmax: bool) -> None:
            super().__init__(flow_model)
            self.flow_model = flow_model
            self.vocab_size = vocab_size
            self.force_argmax = force_argmax

        def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
            state_context = extras["state_context"]
            frame_valid_mask_local = extras["frame_valid_mask"]
            target_frame_index_local = extras["target_frame_index"]
            autocast_enabled_local = bool(extras.get("autocast_enabled", False))

            bsz, fcount, hh, ww = state_context.shape
            target_tokens = x.view(bsz, hh, ww).long()
            state = state_context.clone()
            state[batch_idx, target_frame_index_local] = target_tokens

            frame_times = torch.zeros((bsz, fcount), dtype=torch.float32, device=state.device)
            frame_times[batch_idx, target_frame_index_local] = t

            state_onehot = one_hot_frames(state, num_colors=self.vocab_size)
            cudagraph_step_begin_if_available()
            with autocast_context(state.device, autocast_enabled_local):
                logits = self.flow_model(state_onehot, frame_times, frame_valid_mask=frame_valid_mask_local)
            target_logits = logits[batch_idx, target_frame_index_local].float().reshape(bsz, hh * ww, self.vocab_size)
            probs = torch.softmax(target_logits, dim=-1)
            if self.force_argmax:
                argmax_tokens = probs.argmax(dim=-1)
                probs = F.one_hot(argmax_tokens, num_classes=self.vocab_size).float()
            return probs

    wrapper = ARCPosteriorWrapper(model, num_colors, force_argmax=(reverse_sampler == "argmax"))
    solver = MixtureDiscreteEulerSolver(model=wrapper, path=path, vocabulary_size=num_colors)

    x_init = torch.randint(
        low=0,
        high=num_colors,
        size=(batch_size, height * width),
        device=device,
        dtype=torch.long,
    )
    sampled = solver.sample(
        x_init=x_init,
        step_size=step_size,
        time_grid=torch.tensor([0.0, 1.0], device=device),
        return_intermediates=False,
        verbose=False,
        state_context=frames,
        frame_valid_mask=frame_valid_mask,
        target_frame_index=target_frame_index,
        autocast_enabled=autocast_enabled,
    )
    predicted = sampled.view(batch_size, height, width)
    original_target = frames[batch_idx, target_frame_index]
    return torch.where(target_valid_mask, predicted, original_target)


@torch.no_grad()
def evaluate_last_frame_accuracy(
    model: ARCFlowViT,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    num_colors: int,
    sample_steps: int,
    path: MixtureDiscreteProbPath,
    reverse_sampler: str,
    collect_examples: int = 0,
    show_progress: bool = True,
    autocast_enabled: bool = False,
) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
    if loader is None:
        return {
            "sample_acc": 0.0,
            "sample_acc_at_50": 0.0,
            "sample_acc_at_80": 0.0,
            "sample_acc_at_90": 0.0,
            "sample_acc_at_95": 0.0,
            "task_acc": 0.0,
            "samples": 0.0,
        }, []

    episode_results: Dict[str, tuple[str, bool, bool, bool, bool]] = {}
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
            path=path,
            reverse_sampler=reverse_sampler,
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
    }, examples


@torch.no_grad()
def evaluate_discrete_flow_loss(
    model: ARCFlowViT,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    num_colors: int,
    time_discretization_steps: int,
    path: MixtureDiscreteProbPath,
    loss_function: _Loss,
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
        cudagraph_step_begin_if_available()
        with autocast_context(device, autocast_enabled):
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

    model_cls = ARCFlowViTLooped if args.model_arch == "flow_vit_looped" else ARCFlowViT
    model_loops = args.n_loops if args.model_arch == "flow_vit_looped" else 1
    if is_main and args.verbose:
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
    path = build_discrete_path(
        scheduler_name=args.discrete_scheduler,
        discrete_rate=args.discrete_rate,
        discrete_poly_n=args.discrete_poly_n,
        discrete_vp_beta_min=args.discrete_vp_beta_min,
        discrete_vp_beta_max=args.discrete_vp_beta_max,
    )
    loss_function = get_loss_function(args.loss_function, path=path)
    if is_main and args.verbose:
        if args.discrete_scheduler == "exponential":
            scheduler_desc = f"beta={args.discrete_rate}"
        elif args.discrete_scheduler == "polynomial":
            scheduler_desc = f"n={args.discrete_poly_n}"
        elif args.discrete_scheduler == "vp":
            scheduler_desc = f"beta_min={args.discrete_vp_beta_min}, beta_max={args.discrete_vp_beta_max}"
        else:
            scheduler_desc = ""
        print(f"Discrete scheduler: {args.discrete_scheduler}" + (f" ({scheduler_desc})" if scheduler_desc else ""))
        print(f"Discrete loss: {args.loss_function}")
        print(f"Train/eval-loss time discretization steps: {args.train_time_discretization_steps}")

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
                valid_mask_bool = frame_valid_mask.bool()
                changed_mask = (x_t_tokens != frames) & valid_mask_bool
                noise_all = changed_mask.float().sum() / valid_mask_bool.float().sum().clamp_min(1.0)
                batch_ids = torch.arange(batch_size, device=device)
                target_valid = target_valid_mask.bool()
                target_changed = changed_mask[batch_ids, target_frame_index]
                noise_target = target_changed.float().sum() / target_valid.float().sum().clamp_min(1.0)
                context_valid = valid_mask_bool.clone()
                context_valid[batch_ids, target_frame_index] = False
                noise_context = (changed_mask.float() * context_valid.float()).sum() / context_valid.float().sum().clamp_min(1.0)
                log_line = " | ".join(
                    [
                        f"epoch={epoch}",
                        f"step={global_step}",
                        f"batch={batch_idx}/{total_batches}",
                        f"step_loss={loss_value:.6f}",
                        f"step_avg_loss={step_avg_loss:.6f}",
                        f"noise_all={float(noise_all):.3f}",
                        f"noise_ctx={float(noise_context):.3f}",
                        f"noise_tgt={float(noise_target):.3f}",
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
                            "train/noise_fraction_all": float(noise_all),
                            "train/noise_fraction_context": float(noise_context),
                            "train/noise_fraction_target": float(noise_target),
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
                            clean_frames=frames[sample_idx].detach().cpu(),
                            noisy_frames=x_t_tokens[sample_idx].detach().cpu(),
                            frame_valid_mask=frame_valid_mask[sample_idx].detach().cpu(),
                            target_frame_index=target_idx,
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
                # Free large per-step training tensors before eval to avoid transient OOM spikes.
                del frames, frame_valid_mask, target_frame_index, target_valid_mask
                del frame_times, x_t_tokens, x_t, logits, loss
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                eval_round += 1
                if eval_sampler is not None:
                    eval_sampler.set_epoch(eval_round)
                eval_loss = evaluate_discrete_flow_loss(
                    model,
                    eval_loader if eval_loader is not None else train_loader,
                    device=device,
                    num_colors=args.num_colors,
                    time_discretization_steps=args.train_time_discretization_steps,
                    path=path,
                    loss_function=loss_function,
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
                    path=path,
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
                time_discretization_steps=args.train_time_discretization_steps,
                path=path,
                loss_function=loss_function,
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
                path=path,
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
