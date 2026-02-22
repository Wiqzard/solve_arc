from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_SEARCH_SPACE: Dict[str, List[Any]] = {
    "learning-rate": [3e-4, 1e-4],
    "discrete-rate": [3.0, 5.0, 7.0],
    "rope-3d": [False, True],
    "rope-base": [256.0, 2048.0],
    "num-heads": [8, 12, 16],
    "depth": [6, 10, 14, 20],
    "embed-dim": [384, 512, 768, 1024],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyperparameter search launcher for flow_train_discrete_ARC.py",
    )
    parser.add_argument("--mode", type=str, default="grid", choices=("random", "grid"))
    parser.add_argument("--num-trials", type=int, default=12, help="Used in random mode.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--omp-num-threads", type=int, default=8)
    parser.add_argument("--workspace", type=str, default=".", help="Repo root where training script is located.")
    parser.add_argument("--output-dir", type=str, default="sweeps/flow_discrete_hparam")
    parser.add_argument("--metric-key", type=str, default="eval_loss", help="Metric key to rank trials by.")
    parser.add_argument("--metric-goal", type=str, default="minimize", choices=("minimize", "maximize"))

    parser.add_argument(
        "--search-space-file",
        type=str,
        default="",
        help='JSON file with search space dict, e.g. {"learning-rate": [1e-4,2e-4], "depth": [10,14]}',
    )
    parser.add_argument(
        "--search-space-json",
        type=str,
        default="",
        help="Inline JSON dict with search space; overrides defaults when provided.",
    )

    # Baseline config follows script/flow_train_context_vit_discrete_ddp.sh,
    # except epochs defaults to 1 for hparam search.
    parser.add_argument("--data-root", type=str, default="raw_data/ARC-AGI")
    parser.add_argument("--train-split", type=str, default="training")
    parser.add_argument("--eval-split", type=str, default="evaluation")
    parser.add_argument("--max-demos", "--num-demos", dest="max_demos", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-colors", type=int, default=12)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attention-backend", type=str, default="auto", choices=("auto", "flex", "sdpa"))
    parser.add_argument("--rope-base", type=float, default=256.0)
    parser.add_argument("--rope-3d", action="store_true")

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--log-every-steps", type=int, default=5)
    parser.add_argument("--eval-every-steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=("cosine", "none"))
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--bf16-autocast", action="store_true", default=True)
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=("default", "reduce-overhead", "max-autotune"),
    )
    parser.add_argument("--discrete-rate", type=float, default=5.0)
    parser.add_argument("--reverse-sampler", type=str, default="sample", choices=("sample", "argmax"))
    parser.add_argument("--sample-steps", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--use-wandb", action="store_true", default=True)
    parser.add_argument("--wandb-project", type=str, default="VisionARC")
    parser.add_argument("--wandb-run-prefix", type=str, default="flow-context-vit-discrete-hparam")
    parser.add_argument("--wandb-num-vis-samples", type=int, default=8)
    parser.add_argument("--wandb-vis-scale", type=int, default=8)

    parser.add_argument("--include-rearc", action="store_true", default=True)
    parser.add_argument("--rearc-path", type=str, default="raw_data/re_arc")
    parser.add_argument("--rearc-limit", type=int, default=-1)
    parser.add_argument("--include-barc", action="store_true", default=True)
    parser.add_argument("--barc-path", type=str, default="raw_data/BARC")
    parser.add_argument("--barc-limit", type=int, default=400000)

    parser.add_argument("--framewise-causal-attention", action="store_true", default=True)
    parser.add_argument("--flow-train-translation-aug", action="store_true", default=True)
    parser.add_argument("--flow-train-resolution-aug", action="store_true", default=True)
    parser.add_argument("--nested-dropout", action="store_true", default=False)

    parser.add_argument("--extra-args", type=str, default="", help="Extra args appended verbatim to every run.")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=int(os.environ.get("NPROC_PER_NODE", "8")),
        help="GPUs per trial for DDP (torch.distributed.run).",
    )
    return parser.parse_args()


def _load_search_space(args: argparse.Namespace) -> Dict[str, List[Any]]:
    if args.search_space_json:
        payload = json.loads(args.search_space_json)
    elif args.search_space_file:
        with Path(args.search_space_file).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = DEFAULT_SEARCH_SPACE
    if not isinstance(payload, dict):
        raise ValueError("Search space must be a JSON dict of {flag_name: [values...]}.")
    normalized: Dict[str, List[Any]] = {}
    for key, values in payload.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Search space for '{key}' must be a non-empty list.")
        normalized[str(key)] = values
    return normalized


def _is_valid_trial(trial: Dict[str, Any]) -> bool:
    embed_dim = trial.get("embed-dim")
    num_heads = trial.get("num-heads")
    if embed_dim is not None and num_heads is not None:
        try:
            if int(embed_dim) % int(num_heads) != 0:
                return False
        except Exception:
            return False
    return True


def _build_trials(
    *,
    mode: str,
    space: Dict[str, List[Any]],
    num_trials: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    keys = sorted(space.keys())

    if mode == "grid":
        combos = itertools.product(*(space[key] for key in keys))
        all_trials = [{keys[i]: combo[i] for i in range(len(keys))} for combo in combos]
        return [trial for trial in all_trials if _is_valid_trial(trial)]

    wanted_trials = max(num_trials, 1)
    trials: List[Dict[str, Any]] = []
    seen_signatures = set()
    attempts = 0
    max_attempts = max(wanted_trials * 100, 200)
    while len(trials) < wanted_trials and attempts < max_attempts:
        attempts += 1
        trial = {key: rng.choice(space[key]) for key in keys}
        if not _is_valid_trial(trial):
            continue
        signature = json.dumps(trial, sort_keys=True)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        trials.append(trial)

    if len(trials) < wanted_trials:
        raise RuntimeError(
            f"Could not sample {wanted_trials} unique valid trials after {attempts} attempts; "
            "reduce num-trials or relax the search space."
        )
    return trials


def _metric_from_checkpoint(path: Path, metric_key: str) -> float:
    if not path.exists():
        return float("nan")
    try:
        import torch
    except ImportError:
        return float("nan")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    if not isinstance(metrics, dict):
        return float("nan")
    candidates = [
        metric_key,
        f"eval_{metric_key}",
        metric_key.replace("/", "_"),
        metric_key.replace("_", "/"),
        "eval_loss",
        "eval/loss",
        "task_acc",
        "eval_task_acc",
        "sample_acc",
        "eval_sample_acc",
    ]
    for key in candidates:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return float("nan")


def _stream_run(command: List[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=dict(os.environ),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def _extend_bool_flag(cmd: List[str], condition: bool, flag: str) -> None:
    if condition:
        cmd.append(flag)


def _append_trial_flags(cmd: List[str], params: Dict[str, Any]) -> None:
    for key, value in params.items():
        flag = key if key.startswith("--") else f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])


def main() -> None:
    args = parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.omp_num_threads)

    workspace = Path(args.workspace).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    search_space = _load_search_space(args)
    trials = _build_trials(mode=args.mode, space=search_space, num_trials=args.num_trials, seed=args.seed)
    if not trials:
        raise RuntimeError("No trials generated.")

    print(f"Generated {len(trials)} valid trials (mode={args.mode}).")

    results_jsonl = output_dir / "results.jsonl"
    results_csv = output_dir / "results.csv"

    run_rows: List[Dict[str, Any]] = []
    best_row: Dict[str, Any] | None = None
    best_score = float("inf") if args.metric_goal == "minimize" else float("-inf")

    for trial_index, params in enumerate(trials, start=1):
        trial_name = f"trial_{trial_index:04d}"
        trial_dir = output_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        save_path = trial_dir / "checkpoint_last.pt"
        best_save_path = trial_dir / "checkpoint_best.pt"
        log_path = trial_dir / "train.log"

        cmd: List[str] = [
            args.python_bin,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(args.nproc_per_node),
            "flow_train_discrete_ARC.py",
            "--ddp",
            "--data-root",
            args.data_root,
            "--train-split",
            args.train_split,
            "--eval-split",
            args.eval_split,
            "--max-demos",
            str(args.max_demos),
            "--image-size",
            str(args.image_size),
            "--num-colors",
            str(args.num_colors),
            "--embed-dim",
            str(args.embed_dim),
            "--depth",
            str(args.depth),
            "--num-heads",
            str(args.num_heads),
            "--mlp-ratio",
            str(args.mlp_ratio),
            "--dropout",
            str(args.dropout),
            "--attention-backend",
            args.attention_backend,
            "--rope-base",
            str(args.rope_base),
            "--batch-size",
            str(args.batch_size),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--log-every-steps",
            str(args.log_every_steps),
            "--eval-every-steps",
            str(args.eval_every_steps),
            "--epochs",
            str(args.epochs),
            "--learning-rate",
            str(args.learning_rate),
            "--lr-scheduler",
            args.lr_scheduler,
            "--min-learning-rate",
            str(args.min_learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--discrete-rate",
            str(args.discrete_rate),
            "--reverse-sampler",
            args.reverse_sampler,
            "--sample-steps",
            str(args.sample_steps),
            "--num-workers",
            str(args.num_workers),
            "--save-path",
            str(save_path),
            "--best-save-path",
            str(best_save_path),
        ]
        _extend_bool_flag(cmd, args.include_rearc, "--include-rearc")
        if args.include_rearc:
            cmd.extend(["--rearc-path", args.rearc_path, "--rearc-limit", str(args.rearc_limit)])
        _extend_bool_flag(cmd, args.include_barc, "--include-barc")
        if args.include_barc:
            cmd.extend(["--barc-path", args.barc_path, "--barc-limit", str(args.barc_limit)])
        _extend_bool_flag(cmd, args.bf16_autocast, "--bf16-autocast")
        _extend_bool_flag(cmd, args.compile, "--compile")
        if args.compile:
            cmd.extend(["--compile-mode", args.compile_mode])
        _extend_bool_flag(cmd, args.framewise_causal_attention, "--framewise-causal-attention")
        _extend_bool_flag(cmd, args.rope_3d, "--rope-3d")
        _extend_bool_flag(cmd, args.flow_train_translation_aug, "--flow-train-translation-aug")
        _extend_bool_flag(cmd, args.flow_train_resolution_aug, "--flow-train-resolution-aug")
        _extend_bool_flag(cmd, args.nested_dropout, "--nested-dropout")
        _extend_bool_flag(cmd, args.use_wandb, "--use-wandb")
        if args.use_wandb:
            cmd.extend(
                [
                    "--wandb-project",
                    args.wandb_project,
                    "--wandb-run-name",
                    f"{args.wandb_run_prefix}-{trial_name}",
                    "--wandb-num-vis-samples",
                    str(args.wandb_num_vis_samples),
                    "--wandb-vis-scale",
                    str(args.wandb_vis_scale),
                ]
            )

        _append_trial_flags(cmd, params)
        if args.extra_args:
            cmd.extend(shlex.split(args.extra_args))

        print("=" * 90)
        print(f"[{trial_index}/{len(trials)}] {trial_name}")
        print("Command:", " ".join(shlex.quote(x) for x in cmd))
        start_time = time.time()
        return_code = _stream_run(cmd, cwd=workspace, log_path=log_path)
        duration_sec = time.time() - start_time

        score = _metric_from_checkpoint(best_save_path, args.metric_key)
        if score != score:  # NaN fallback
            score = _metric_from_checkpoint(save_path, args.metric_key)

        row = {
            "trial_name": trial_name,
            "return_code": return_code,
            "duration_sec": round(duration_sec, 3),
            "metric": score,
            "checkpoint_best": str(best_save_path),
            "checkpoint_last": str(save_path),
            "params": params,
            "log": str(log_path),
        }
        run_rows.append(row)

        with results_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

        if return_code == 0 and score == score:
            is_better = score < best_score if args.metric_goal == "minimize" else score > best_score
            if is_better:
                best_score = score
                best_row = row

        print(f"Trial {trial_name} finished: rc={return_code} metric={score} duration={duration_sec:.1f}s")

    with results_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["trial_name", "return_code", "duration_sec", "metric", "checkpoint_best", "checkpoint_last", "log", "params"],
        )
        writer.writeheader()
        for row in run_rows:
            csv_row = dict(row)
            csv_row["params"] = json.dumps(row["params"], ensure_ascii=True, sort_keys=True)
            writer.writerow(csv_row)

    print("=" * 90)
    print(f"Completed {len(run_rows)} trials. Results:")
    print(f"- JSONL: {results_jsonl}")
    print(f"- CSV:   {results_csv}")
    if best_row is None:
        print("No successful trial produced a valid metric.")
    else:
        print(f"Best trial: {best_row['trial_name']} metric={best_row['metric']}")
        print(f"Best params: {json.dumps(best_row['params'], ensure_ascii=True, sort_keys=True)}")


if __name__ == "__main__":
    main()
