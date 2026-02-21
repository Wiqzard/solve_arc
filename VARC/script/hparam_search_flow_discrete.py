from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

DEFAULT_SEARCH_SPACE: Dict[str, List[Any]] = {
    "learning-rate": [1e-4, 2e-4, 3e-4],
    "depth": [10, 14],
    "discrete-rate": [3.0, 5.0, 7.0],
    "sample-steps": [20, 40],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyperparameter search launcher for flow_train_discrete_ARC.py",
    )
    parser.add_argument("--mode", type=str, default="random", choices=("random", "grid"))
    parser.add_argument("--num-trials", type=int, default=12, help="Used in random mode.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workspace", type=str, default=".", help="Repo root where training script is located.")
    parser.add_argument("--output-dir", type=str, default="sweeps/flow_discrete_hparam")
    parser.add_argument("--metric-key", type=str, default="task_acc", help="Metric key to rank trials by.")

    parser.add_argument(
        "--search-space-file",
        type=str,
        default="",
        help="JSON file with search space dict, e.g. {\"learning-rate\": [1e-4,2e-4], \"depth\": [10,14]}",
    )
    parser.add_argument(
        "--search-space-json",
        type=str,
        default="",
        help="Inline JSON dict with search space; overrides defaults when provided.",
    )

    parser.add_argument("--data-root", type=str, default="raw_data/ARC-AGI")
    parser.add_argument("--train-split", type=str, default="training")
    parser.add_argument("--eval-split", type=str, default="evaluation")
    parser.add_argument("--num-demos", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=30)
    parser.add_argument("--num-colors", type=int, default=12)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--log-every-steps", type=int, default=5)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=("cosine", "none"))
    parser.add_argument("--reverse-sampler", type=str, default="sample", choices=("sample", "argmax"))
    parser.add_argument("--attention-backend", type=str, default="auto", choices=("auto", "flex", "sdpa"))

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="VisionARC")
    parser.add_argument("--wandb-run-prefix", type=str, default="flow-discrete-hparam")

    parser.add_argument("--include-rearc", action="store_true")
    parser.add_argument("--rearc-path", type=str, default="raw_data/re_arc")
    parser.add_argument("--rearc-limit", type=int, default=-1)
    parser.add_argument("--include-barc", action="store_true")
    parser.add_argument("--barc-path", type=str, default="raw_data/BARC")
    parser.add_argument("--barc-limit", type=int, default=-1)

    parser.add_argument("--bf16-autocast", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead", choices=("default", "reduce-overhead", "max-autotune"))
    parser.add_argument("--framewise-causal-attention", action="store_true")
    parser.add_argument("--flow-train-translation-aug", action="store_true")
    parser.add_argument("--flow-train-resolution-aug", action="store_true")
    parser.add_argument("--extra-args", type=str, default="", help="Extra args appended verbatim to every run.")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
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
        return [{keys[i]: combo[i] for i in range(len(keys))} for combo in combos]
    return [
        {key: rng.choice(space[key]) for key in keys}
        for _ in range(max(num_trials, 1))
    ]


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
    workspace = Path(args.workspace).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    search_space = _load_search_space(args)
    trials = _build_trials(mode=args.mode, space=search_space, num_trials=args.num_trials, seed=args.seed)
    if not trials:
        raise RuntimeError("No trials generated.")

    results_jsonl = output_dir / "results.jsonl"
    results_csv = output_dir / "results.csv"

    run_rows: List[Dict[str, Any]] = []
    best_row: Dict[str, Any] | None = None
    best_score = float("-inf")

    for trial_index, params in enumerate(trials, start=1):
        trial_name = f"trial_{trial_index:04d}"
        trial_dir = output_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        save_path = trial_dir / "checkpoint_last.pt"
        best_save_path = trial_dir / "checkpoint_best.pt"
        log_path = trial_dir / "train.log"

        cmd: List[str] = [
            args.python_bin,
            "flow_train_discrete_ARC.py",
            "--data-root",
            args.data_root,
            "--train-split",
            args.train_split,
            "--eval-split",
            args.eval_split,
            "--num-demos",
            str(args.num_demos),
            "--image-size",
            str(args.image_size),
            "--num-colors",
            str(args.num_colors),
            "--embed-dim",
            str(args.embed_dim),
            "--num-heads",
            str(args.num_heads),
            "--mlp-ratio",
            str(args.mlp_ratio),
            "--dropout",
            str(args.dropout),
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
            "--weight-decay",
            str(args.weight_decay),
            "--lr-scheduler",
            args.lr_scheduler,
            "--reverse-sampler",
            args.reverse_sampler,
            "--attention-backend",
            args.attention_backend,
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
        _extend_bool_flag(cmd, args.flow_train_translation_aug, "--flow-train-translation-aug")
        _extend_bool_flag(cmd, args.flow_train_resolution_aug, "--flow-train-resolution-aug")
        _extend_bool_flag(cmd, args.use_wandb, "--use-wandb")
        if args.use_wandb:
            cmd.extend(
                [
                    "--wandb-project",
                    args.wandb_project,
                    "--wandb-run-name",
                    f"{args.wandb_run_prefix}-{trial_name}",
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

        if return_code == 0 and score == score and score > best_score:
            best_score = score
            best_row = row

        print(
            f"Trial {trial_name} finished: rc={return_code} metric={score} duration={duration_sec:.1f}s"
        )

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
