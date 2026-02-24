#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class LifeLikeRule:
    name: str
    birth: Tuple[int, ...]
    survive: Tuple[int, ...]
    family: str


RULE_LIBRARY: Dict[str, LifeLikeRule] = {
    "life": LifeLikeRule(name="life", birth=(3,), survive=(2, 3), family="complex"),
    "highlife": LifeLikeRule(name="highlife", birth=(3, 6), survive=(2, 3), family="complex"),
    "day_night": LifeLikeRule(
        name="day_night",
        birth=(3, 6, 7, 8),
        survive=(3, 4, 6, 7, 8),
        family="chaotic",
    ),
    "seeds": LifeLikeRule(name="seeds", birth=(2,), survive=(), family="chaotic"),
    "replicator": LifeLikeRule(name="replicator", birth=(1, 3, 5, 7), survive=(1, 3, 5, 7), family="chaotic"),
    "anneal": LifeLikeRule(name="anneal", birth=(4, 6, 7, 8), survive=(3, 5, 6, 7, 8), family="ordered"),
    "maze": LifeLikeRule(name="maze", birth=(3,), survive=(1, 2, 3, 4, 5), family="ordered"),
    "coagulations": LifeLikeRule(name="coagulations", birth=(3, 7, 8), survive=(2, 3, 5, 6, 7, 8), family="ordered"),
    "morley": LifeLikeRule(name="morley", birth=(3, 6, 8), survive=(2, 4, 5), family="complex"),
}


RULE_PROFILES: Dict[str, Tuple[str, ...]] = {
    "all": tuple(RULE_LIBRARY.keys()),
    "ordered": tuple(name for name, rule in RULE_LIBRARY.items() if rule.family == "ordered"),
    "complex": tuple(name for name, rule in RULE_LIBRARY.items() if rule.family == "complex"),
    "chaotic": tuple(name for name, rule in RULE_LIBRARY.items() if rule.family == "chaotic"),
    "paper_like": (
        "maze",
        "anneal",
        "life",
        "highlife",
        "morley",
        "day_night",
        "replicator",
        "seeds",
    ),
}


def parse_float_list(text: str) -> List[float]:
    values = [v.strip() for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return [float(v) for v in values]


def parse_int_list(text: str) -> List[int]:
    values = [v.strip() for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return [int(v) for v in values]


def parse_rule_names(rule_names_text: str | None, profile: str) -> List[LifeLikeRule]:
    if rule_names_text:
        names = [name.strip() for name in rule_names_text.split(",") if name.strip()]
    else:
        if profile not in RULE_PROFILES:
            valid = ", ".join(sorted(RULE_PROFILES))
            raise ValueError(f"Unknown rule profile: {profile}. Valid profiles: {valid}")
        names = list(RULE_PROFILES[profile])

    rules: List[LifeLikeRule] = []
    for name in names:
        if name not in RULE_LIBRARY:
            valid = ", ".join(sorted(RULE_LIBRARY))
            raise ValueError(f"Unknown rule: {name}. Valid rules: {valid}")
        rules.append(RULE_LIBRARY[name])
    return rules


def build_lookup_mask(active_values: Iterable[int]) -> np.ndarray:
    mask = np.zeros(9, dtype=np.bool_)
    for value in active_values:
        if value < 0 or value > 8:
            raise ValueError(f"Life-like neighborhood counts must be in [0, 8], got {value}.")
        mask[value] = True
    return mask


def ca_step_binary(state: np.ndarray, birth_mask: np.ndarray, survive_mask: np.ndarray) -> np.ndarray:
    neighbors = (
        np.roll(state, shift=(1, 0), axis=(0, 1))
        + np.roll(state, shift=(-1, 0), axis=(0, 1))
        + np.roll(state, shift=(0, 1), axis=(0, 1))
        + np.roll(state, shift=(0, -1), axis=(0, 1))
        + np.roll(state, shift=(1, 1), axis=(0, 1))
        + np.roll(state, shift=(1, -1), axis=(0, 1))
        + np.roll(state, shift=(-1, 1), axis=(0, 1))
        + np.roll(state, shift=(-1, -1), axis=(0, 1))
    )
    born = (state == 0) & birth_mask[neighbors]
    survives = (state == 1) & survive_mask[neighbors]
    return (born | survives).astype(np.uint8)


def rollout_rule(
    *,
    rule: LifeLikeRule,
    sim_size: int,
    rollout_steps: int,
    init_density: float,
    rng: np.random.Generator,
) -> np.ndarray:
    birth_mask = build_lookup_mask(rule.birth)
    survive_mask = build_lookup_mask(rule.survive)

    state = (rng.random((sim_size, sim_size)) < init_density).astype(np.uint8)
    frames = np.empty((rollout_steps + 1, sim_size, sim_size), dtype=np.uint8)
    frames[0] = state
    for step in range(1, rollout_steps + 1):
        state = ca_step_binary(state, birth_mask, survive_mask)
        frames[step] = state
    return frames


def sample_io_pair(
    *,
    frames: np.ndarray,
    crop_size: int,
    horizon: int,
    warmup_steps: int,
    rng: random.Random,
) -> Tuple[List[List[int]], List[List[int]], int, int, int]:
    max_t = frames.shape[0] - 1 - horizon
    if max_t < warmup_steps:
        raise ValueError(
            f"rollout_steps ({frames.shape[0]-1}) too small for horizon={horizon} and warmup={warmup_steps}."
        )
    t = rng.randint(warmup_steps, max_t)

    sim_h, sim_w = int(frames.shape[1]), int(frames.shape[2])
    if crop_size > sim_h or crop_size > sim_w:
        raise ValueError(f"crop_size={crop_size} must be <= sim_size={sim_h}.")
    y0 = rng.randint(0, sim_h - crop_size)
    x0 = rng.randint(0, sim_w - crop_size)

    x_t = frames[t, y0 : y0 + crop_size, x0 : x0 + crop_size]
    x_tp = frames[t + horizon, y0 : y0 + crop_size, x0 : x0 + crop_size]
    return x_t.tolist(), x_tp.tolist(), t, y0, x0


def make_task_payload(
    *,
    task_id: str,
    rule: LifeLikeRule,
    init_density: float,
    horizon: int,
    sim_size: int,
    crop_size: int,
    rollout_steps: int,
    warmup_steps: int,
    num_train_examples: int,
    num_test_examples: int,
    np_rng: np.random.Generator,
    py_rng: random.Random,
) -> Dict[str, object]:
    frames = rollout_rule(
        rule=rule,
        sim_size=sim_size,
        rollout_steps=rollout_steps,
        init_density=init_density,
        rng=np_rng,
    )

    train_examples = []
    for _ in range(num_train_examples):
        inp, out, t, y0, x0 = sample_io_pair(
            frames=frames,
            crop_size=crop_size,
            horizon=horizon,
            warmup_steps=warmup_steps,
            rng=py_rng,
        )
        train_examples.append({"input": inp, "output": out, "meta": {"t": t, "y0": y0, "x0": x0}})

    test_examples = []
    for _ in range(num_test_examples):
        inp, out, t, y0, x0 = sample_io_pair(
            frames=frames,
            crop_size=crop_size,
            horizon=horizon,
            warmup_steps=warmup_steps,
            rng=py_rng,
        )
        test_examples.append({"input": inp, "output": out, "meta": {"t": t, "y0": y0, "x0": x0}})

    return {
        "train": train_examples,
        "test": test_examples,
        "metadata": {
            "task_id": task_id,
            "rule_name": rule.name,
            "rule_family": rule.family,
            "birth": list(rule.birth),
            "survive": list(rule.survive),
            "initial_density": init_density,
            "prediction_horizon": horizon,
            "sim_size": sim_size,
            "crop_size": crop_size,
            "rollout_steps": rollout_steps,
            "warmup_steps": warmup_steps,
        },
    }


def write_split(
    *,
    split_name: str,
    num_tasks: int,
    out_dir: Path,
    rules: Sequence[LifeLikeRule],
    densities: Sequence[float],
    horizons: Sequence[int],
    sim_size: int,
    crop_size: int,
    rollout_steps: int,
    warmup_steps: int,
    num_train_examples: int,
    num_test_examples: int,
    seed: int,
) -> Dict[str, int]:
    split_dir = out_dir / "data" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    configs: List[Tuple[LifeLikeRule, float, int]] = []
    for rule in rules:
        for density in densities:
            for horizon in horizons:
                configs.append((rule, density, horizon))

    if not configs:
        raise ValueError("No CA configurations were constructed.")

    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    py_rng.shuffle(configs)

    config_counts: Dict[str, int] = {}
    for task_idx in range(num_tasks):
        rule, density, horizon = configs[task_idx % len(configs)]
        task_id = f"ca2d_{split_name}_{task_idx:06d}_{rule.name}_h{horizon}_p{int(density*100):02d}"
        payload = make_task_payload(
            task_id=task_id,
            rule=rule,
            init_density=density,
            horizon=horizon,
            sim_size=sim_size,
            crop_size=crop_size,
            rollout_steps=rollout_steps,
            warmup_steps=warmup_steps,
            num_train_examples=num_train_examples,
            num_test_examples=num_test_examples,
            np_rng=np_rng,
            py_rng=py_rng,
        )
        with (split_dir / f"{task_id}.json").open("w") as fh:
            json.dump(payload, fh, separators=(",", ":"))

        key = f"{rule.name}|h{horizon}|p{density:.2f}"
        config_counts[key] = config_counts.get(key, 0) + 1

    return config_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 2D cellular-automata dataset in ARC JSON format for discrete flow-matching pretraining."
    )
    parser.add_argument("--output-root", type=Path, default=Path("raw_data/CA-2D-DFM"))
    parser.add_argument("--train-split", type=str, default="training")
    parser.add_argument("--eval-split", type=str, default="evaluation")
    parser.add_argument("--num-train-tasks", type=int, default=2048)
    parser.add_argument("--num-eval-tasks", type=int, default=256)
    parser.add_argument("--train-examples-per-task", type=int, default=12)
    parser.add_argument("--test-examples-per-task", type=int, default=4)
    parser.add_argument("--sim-size", type=int, default=96)
    parser.add_argument("--crop-size", type=int, default=31)
    parser.add_argument("--rollout-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=60)
    parser.add_argument("--densities", type=str, default="0.15,0.25,0.35,0.45,0.55")
    parser.add_argument("--horizons", type=str, default="1,5")
    parser.add_argument(
        "--rule-profile",
        type=str,
        default="paper_like",
        choices=tuple(sorted(RULE_PROFILES)),
        help="Preset rule collection.",
    )
    parser.add_argument(
        "--rules",
        type=str,
        default=None,
        help="Optional comma-separated override for rules (e.g. 'life,highlife,seeds').",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rules = parse_rule_names(args.rules, args.rule_profile)
    densities = parse_float_list(args.densities)
    horizons = parse_int_list(args.horizons)

    if args.crop_size <= 0:
        raise ValueError("crop-size must be positive.")
    if args.train_examples_per_task < 2:
        raise ValueError("train-examples-per-task must be >= 2 so the loader can sample demos.")
    if args.num_train_tasks <= 0 or args.num_eval_tasks <= 0:
        raise ValueError("num-train-tasks and num-eval-tasks must be positive.")

    out_root: Path = args.output_root
    if out_root.exists() and args.overwrite:
        for split in (args.train_split, args.eval_split):
            split_dir = out_root / "data" / split
            if split_dir.exists():
                for file_path in split_dir.glob("*.json"):
                    file_path.unlink()
    out_root.mkdir(parents=True, exist_ok=True)

    train_counts = write_split(
        split_name=args.train_split,
        num_tasks=args.num_train_tasks,
        out_dir=out_root,
        rules=rules,
        densities=densities,
        horizons=horizons,
        sim_size=args.sim_size,
        crop_size=args.crop_size,
        rollout_steps=args.rollout_steps,
        warmup_steps=args.warmup_steps,
        num_train_examples=args.train_examples_per_task,
        num_test_examples=args.test_examples_per_task,
        seed=args.seed,
    )
    eval_counts = write_split(
        split_name=args.eval_split,
        num_tasks=args.num_eval_tasks,
        out_dir=out_root,
        rules=rules,
        densities=densities,
        horizons=horizons,
        sim_size=args.sim_size,
        crop_size=args.crop_size,
        rollout_steps=args.rollout_steps,
        warmup_steps=args.warmup_steps,
        num_train_examples=args.train_examples_per_task,
        num_test_examples=args.test_examples_per_task,
        seed=args.seed + 17,
    )

    summary = {
        "output_root": str(out_root),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "num_train_tasks": args.num_train_tasks,
        "num_eval_tasks": args.num_eval_tasks,
        "rules": [rule.name for rule in rules],
        "densities": densities,
        "horizons": horizons,
        "rollout_steps": args.rollout_steps,
        "warmup_steps": args.warmup_steps,
        "sim_size": args.sim_size,
        "crop_size": args.crop_size,
        "train_examples_per_task": args.train_examples_per_task,
        "test_examples_per_task": args.test_examples_per_task,
        "train_config_counts": train_counts,
        "eval_config_counts": eval_counts,
        "seed": args.seed,
    }
    summary_path = out_root / "metadata.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    print(f"Wrote CA dataset to {out_root}")
    print(f"Training tasks: {args.num_train_tasks}, Eval tasks: {args.num_eval_tasks}")
    print(f"Rules: {[rule.name for rule in rules]}")
    print(f"Horizons: {horizons}, Densities: {densities}")
    print(f"Metadata: {summary_path}")


if __name__ == "__main__":
    main()
