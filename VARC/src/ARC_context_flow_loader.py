from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

IGNORE_INDEX = 10
PAD_INDEX = 11


def _grid_shape(grid: Sequence[Sequence[int]]) -> Tuple[int, int]:
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    return height, width


class ARCContextFlowDataset(Dataset):
    """Build (demo pairs + query input + query output) episodes for flow matching."""

    def __init__(
        self,
        root: Path,
        split: str,
        mode: str,
        *,
        max_demos: int,
        image_size: int,
        num_colors: int,
        seed: int = 42,
        translation_augmentation: bool = True,
        resolution_augmentation: bool = True,
        nested_dropout: bool = False,
        extra_train_roots: Optional[Iterable[Path]] = None,
        extra_train_limit: Optional[int] = None,
        extra_train_specs: Optional[
            Iterable[Tuple[Path, Optional[int], str] | Tuple[Path, Optional[int], str, str]]
        ] = None,
    ) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError("mode must be 'train' or 'eval'.")
        self.root = Path(root)
        self.split = split
        self.mode = mode
        self.max_demos = max_demos
        self.image_size = image_size
        self.num_colors = num_colors
        if self.num_colors <= PAD_INDEX:
            raise ValueError(
                f"num_colors must be >= {PAD_INDEX + 1} to match offline color semantics "
                f"(IGNORE={IGNORE_INDEX}, PAD={PAD_INDEX}); got {self.num_colors}."
            )
        self.seed = seed
        self.translation_augmentation = translation_augmentation
        self.resolution_augmentation = resolution_augmentation
        self.nested_dropout = nested_dropout

        split_dir = self.root / "data" / split
        files = sorted(split_dir.glob("*.json"))
        if not files:
            raise RuntimeError(f"No task files found under {split_dir}")

        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._train_valid_indices: Dict[str, List[int]] = {}
        self._query_records: List[Dict[str, Any]] = []

        for file_path in files:
            task_name = file_path.stem
            with file_path.open("r") as fh:
                task_payload = json.load(fh)
            train_examples = task_payload.get("train", [])
            test_examples = task_payload.get("test", [])
            train_valid = [idx for idx, ex in enumerate(train_examples) if self._is_valid_example(ex)]
            test_valid = [idx for idx, ex in enumerate(test_examples) if self._is_valid_example(ex)]
            if not train_valid:
                continue

            self._tasks[task_name] = task_payload
            self._train_valid_indices[task_name] = train_valid

            if mode == "train":
                query_source = "train"
                query_indices = train_valid
            else:
                query_source = "test"
                query_indices = test_valid

            for query_index in query_indices:
                self._query_records.append(
                    {
                        "task_name": task_name,
                        "query_source": query_source,
                        "query_index": query_index,
                    }
                )

        if mode == "train":
            if extra_train_roots:
                for extra_root in extra_train_roots:
                    self._load_extra_training_data(
                        extra_root=Path(extra_root),
                        limit_per_task=extra_train_limit,
                        source_name="extra",
                    )
            if extra_train_specs:
                for spec in extra_train_specs:
                    if len(spec) == 3:
                        extra_root, per_task_limit, source_name = spec
                        limit_scope = "per_task"
                    elif len(spec) == 4:
                        extra_root, per_task_limit, source_name, limit_scope = spec
                    else:
                        raise ValueError(
                            "extra_train_specs entries must be (path, limit, source) "
                            "or (path, limit, source, limit_scope)."
                        )
                    self._load_extra_training_data(
                        extra_root=Path(extra_root),
                        limit_per_task=per_task_limit,
                        source_name=source_name,
                        limit_scope=str(limit_scope),
                    )

        if not self._query_records:
            raise RuntimeError(f"No valid episodes found for split={split} mode={mode}.")

    def __len__(self) -> int:
        return len(self._query_records)

    def _is_valid_grid(self, grid: Sequence[Sequence[int]]) -> bool:
        if not isinstance(grid, list) or not grid:
            return False
        height, width = _grid_shape(grid)
        if height <= 0 or width <= 0 or height > self.image_size or width > self.image_size:
            return False
        arr = np.asarray(grid, dtype=np.int64)
        if arr.ndim != 2:
            return False
        if arr.min() < 0 or arr.max() >= self.num_colors:
            return False
        return True

    def _is_valid_example(self, example: Dict[str, Any]) -> bool:
        if "input" not in example or "output" not in example:
            return False
        if not self._is_valid_grid(example["input"]) or not self._is_valid_grid(example["output"]):
            return False
        output_h, output_w = _grid_shape(example["output"])
        # Output frames add an explicit border token (+1 row/col), like offline_train.
        if output_h + 1 > self.image_size or output_w + 1 > self.image_size:
            return False
        return True

    def _pad_grid(
        self,
        grid: Sequence[Sequence[int]],
        *,
        x_offset: int = 0,
        y_offset: int = 0,
        output_shape: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        canvas = torch.full((self.image_size, self.image_size), IGNORE_INDEX, dtype=torch.long)
        mask = torch.zeros((self.image_size, self.image_size), dtype=torch.bool)
        array = torch.tensor(grid, dtype=torch.long)
        height, width = array.shape
        canvas[y_offset : y_offset + height, x_offset : x_offset + width] = array
        mask[y_offset : y_offset + height, x_offset : x_offset + width] = True
        if output_shape:
            canvas[y_offset : y_offset + height, x_offset + width] = PAD_INDEX
            canvas[y_offset + height, x_offset : x_offset + width + 1] = PAD_INDEX
            mask[y_offset : y_offset + height + 1, x_offset : x_offset + width + 1] = True
        return canvas, mask

    def _augment_example_pair(
        self,
        example: Dict[str, Any],
        *,
        rng: random.Random | Any,
    ) -> Dict[str, List[List[int]]]:
        input_grid = np.asarray(example["input"], dtype=np.int64)
        output_grid = np.asarray(example["output"], dtype=np.int64)
        input_h, input_w = int(input_grid.shape[0]), int(input_grid.shape[1])
        output_h, output_w = int(output_grid.shape[0]), int(output_grid.shape[1])

        scale_factor = 1
        if self.mode == "train" and self.resolution_augmentation:
            # Match offline semantics:
            # - inputs must fit as-is
            # - outputs reserve one extra border token (+1 row/col)
            max_scale_from_input = min(
                self.image_size // max(input_h, 1),
                self.image_size // max(input_w, 1),
            )
            max_scale_from_output = min(
                (self.image_size - 1) // max(output_h, 1),
                (self.image_size - 1) // max(output_w, 1),
            )
            max_scale_factor = max(min(max_scale_from_input, max_scale_from_output), 1)
            scale_factor = int(rng.randint(1, max_scale_factor))
            if scale_factor > 1:
                input_grid = np.repeat(np.repeat(input_grid, scale_factor, axis=0), scale_factor, axis=1)
                output_grid = np.repeat(np.repeat(output_grid, scale_factor, axis=0), scale_factor, axis=1)

        required_h = max(int(input_grid.shape[0]), int(output_grid.shape[0]) + 1)
        required_w = max(int(input_grid.shape[1]), int(output_grid.shape[1]) + 1)

        x_offset = 0
        y_offset = 0
        if self.mode == "train" and self.translation_augmentation:
            max_x_offset = max(self.image_size - required_w, 0)
            max_y_offset = max(self.image_size - required_h, 0)
            x_offset = int(rng.randint(0, max_x_offset)) if max_x_offset > 0 else 0
            y_offset = int(rng.randint(0, max_y_offset)) if max_y_offset > 0 else 0

        return {
            "input": input_grid.tolist(),
            "output": output_grid.tolist(),
            "x_offset": x_offset,
            "y_offset": y_offset,
        }

    def _select_demo_indices(
        self,
        *,
        task_name: str,
        query_source: str,
        query_index: int,
        rng: random.Random,
    ) -> List[int]:
        candidates = list(self._train_valid_indices[task_name])
        if query_source == "train" and query_index in candidates:
            candidates.remove(query_index)
        if not candidates:
            return []
        if len(candidates) <= self.max_demos:
            selected = list(candidates)
            rng.shuffle(selected)
            return selected
        return rng.sample(candidates, k=self.max_demos)

    def _iter_task_examples_from_payload(
        self,
        payload: Any,
        *,
        default_task_name: str,
    ) -> List[Tuple[str, List[Dict[str, Any]]]]:
        task_sets: List[Tuple[str, List[Dict[str, Any]]]] = []
        if isinstance(payload, list):
            task_sets.append((default_task_name, payload))
            return task_sets
        if not isinstance(payload, dict):
            return task_sets

        # Standard ARC task file: {"train": [...], "test": [...]}
        if isinstance(payload.get("train"), list):
            task_sets.append((default_task_name, payload["train"]))
            return task_sets

        # Container format: {"tasks": {"id": {"train":[...]}, ...}}
        if isinstance(payload.get("tasks"), dict):
            for task_name, task_payload in payload["tasks"].items():
                if isinstance(task_payload, list):
                    task_sets.append((str(task_name), task_payload))
                elif isinstance(task_payload, dict):
                    examples = task_payload.get("train")
                    if not isinstance(examples, list):
                        examples = task_payload.get("examples")
                    if isinstance(examples, list):
                        task_sets.append((str(task_name), examples))
            if task_sets:
                return task_sets

        # Flat mapping format: {"id": {"train":[...]}, "id2": [...], ...}
        for task_name, task_payload in payload.items():
            if isinstance(task_payload, list):
                task_sets.append((str(task_name), task_payload))
                continue
            if isinstance(task_payload, dict):
                examples = task_payload.get("train")
                if not isinstance(examples, list):
                    examples = task_payload.get("examples")
                if isinstance(examples, list):
                    task_sets.append((str(task_name), examples))
        return task_sets

    def _load_extra_training_data(
        self,
        *,
        extra_root: Path,
        limit_per_task: Optional[int],
        source_name: str,
        limit_scope: str = "per_task",
    ) -> None:
        if limit_scope not in {"per_task", "total"}:
            raise ValueError("limit_scope must be one of: per_task, total")
        files: List[Path] = []
        source_location = str(extra_root)

        if extra_root.is_file() and extra_root.suffix == ".json":
            files = [extra_root]
        else:
            candidates = [
                extra_root / "tasks",
                extra_root / "training",
                extra_root / "data" / "training",
                extra_root,
            ]
            for candidate in candidates:
                if candidate.exists() and candidate.is_dir():
                    candidate_files = sorted(candidate.glob("*.json"))
                    if candidate_files:
                        files = candidate_files
                        source_location = str(candidate)
                        break

        if not files:
            print(f"No {source_name} task files found under {extra_root}, skipping.")
            return

        rng = random.Random(42)
        added_queries = 0
        added_per_task: Dict[str, int] = {}
        stop_loading = False
        for file_path in files:
            if stop_loading:
                break
            with file_path.open("r") as fh:
                payload = json.load(fh)

            for task_name, examples in self._iter_task_examples_from_payload(
                payload,
                default_task_name=file_path.stem,
            ):
                valid_examples = [ex for ex in examples if self._is_valid_example(ex)]
                if not valid_examples:
                    continue

                if limit_per_task is not None:
                    if limit_scope == "per_task":
                        remaining = limit_per_task - added_per_task.get(task_name, 0)
                    else:
                        remaining = limit_per_task - added_queries
                    if remaining <= 0:
                        if limit_scope == "total":
                            stop_loading = True
                            break
                        continue
                    rng.shuffle(valid_examples)
                    valid_examples = valid_examples[:remaining]

                if task_name not in self._tasks:
                    self._tasks[task_name] = {"train": [], "test": []}
                    self._train_valid_indices[task_name] = []
                elif "train" not in self._tasks[task_name]:
                    self._tasks[task_name]["train"] = []

                train_examples = self._tasks[task_name]["train"]
                start_index = len(train_examples)
                train_examples.extend(valid_examples)

                new_indices = list(range(start_index, start_index + len(valid_examples)))
                self._train_valid_indices[task_name].extend(new_indices)
                added_per_task[task_name] = added_per_task.get(task_name, 0) + len(valid_examples)

                for query_index in new_indices:
                    self._query_records.append(
                        {
                            "task_name": task_name,
                            "query_source": "train",
                            "query_index": query_index,
                        }
                    )
                    added_queries += 1

        limit_desc = "all" if limit_per_task is None else f"{limit_per_task} ({limit_scope})"
        print(f"Added {added_queries} {source_name} train queries from {source_location} [limit={limit_desc}].")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self._query_records[idx]
        task_name = record["task_name"]
        query_source = record["query_source"]
        query_index = record["query_index"]

        payload = self._tasks[task_name]
        train_examples = payload["train"]
        query_examples = payload["train"] if query_source == "train" else payload["test"]
        query_example = query_examples[query_index]

        rng = random if self.mode == "train" else random.Random(self.seed + idx)
        demo_indices = self._select_demo_indices(
            task_name=task_name,
            query_source=query_source,
            query_index=query_index,
            rng=rng,
        )
        requested_demo_count = self.max_demos
        if self.mode == "train" and self.nested_dropout and self.max_demos > 0:
            requested_demo_count = int(rng.randint(1, self.max_demos))
        keep_demo_count = min(requested_demo_count, len(demo_indices))
        demo_indices = demo_indices[-keep_demo_count:] if keep_demo_count > 0 else []
        pad_demo_count = self.max_demos - keep_demo_count

        frames: List[torch.Tensor] = []
        frame_masks: List[torch.Tensor] = []

        if pad_demo_count > 0:
            pad_frame = torch.full((self.image_size, self.image_size), IGNORE_INDEX, dtype=torch.long)
            pad_mask = torch.zeros((self.image_size, self.image_size), dtype=torch.bool)
            for _ in range(pad_demo_count):
                frames.extend([pad_frame.clone(), pad_frame.clone()])
                frame_masks.extend([pad_mask.clone(), pad_mask.clone()])

        for demo_idx in demo_indices:
            demo = train_examples[demo_idx]
            aug_demo = self._augment_example_pair(demo, rng=rng)
            x_frame, x_mask = self._pad_grid(
                aug_demo["input"],
                x_offset=aug_demo["x_offset"],
                y_offset=aug_demo["y_offset"],
                output_shape=False,
            )
            y_frame, y_mask = self._pad_grid(
                aug_demo["output"],
                x_offset=aug_demo["x_offset"],
                y_offset=aug_demo["y_offset"],
                output_shape=True,
            )
            frames.extend([x_frame, y_frame])
            frame_masks.extend([x_mask, y_mask])

        aug_query = self._augment_example_pair(query_example, rng=rng)
        query_input_frame, query_input_mask = self._pad_grid(
            aug_query["input"],
            x_offset=aug_query["x_offset"],
            y_offset=aug_query["y_offset"],
            output_shape=False,
        )
        query_output_frame, query_output_mask = self._pad_grid(
            aug_query["output"],
            x_offset=aug_query["x_offset"],
            y_offset=aug_query["y_offset"],
            output_shape=True,
        )
        frames.extend([query_input_frame, query_output_frame])
        frame_masks.extend([query_input_mask, query_output_mask])

        stacked_frames = torch.stack(frames, dim=0)  # (F, H, W)
        stacked_masks = torch.stack(frame_masks, dim=0)  # (F, H, W)
        target_frame_index = stacked_frames.size(0) - 1

        return {
            "frames": stacked_frames,
            "frame_valid_mask": stacked_masks,
            "target_frame_index": torch.tensor(target_frame_index, dtype=torch.long),
            "target_output": query_output_frame,
            "target_valid_mask": query_output_mask,
            "active_demo_count": torch.tensor(keep_demo_count, dtype=torch.long),
            "task_name": task_name,
            "query_index": torch.tensor(query_index, dtype=torch.long),
        }


def collate_flow_context(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    frames = torch.stack([item["frames"] for item in batch], dim=0)
    frame_valid_mask = torch.stack([item["frame_valid_mask"] for item in batch], dim=0)
    target_frame_index = torch.stack([item["target_frame_index"] for item in batch], dim=0)
    target_output = torch.stack([item["target_output"] for item in batch], dim=0)
    target_valid_mask = torch.stack([item["target_valid_mask"] for item in batch], dim=0)
    active_demo_count = torch.stack([item["active_demo_count"] for item in batch], dim=0)
    task_names = [item["task_name"] for item in batch]
    query_indices = torch.stack([item["query_index"] for item in batch], dim=0)

    return {
        "frames": frames,
        "frame_valid_mask": frame_valid_mask,
        "target_frame_index": target_frame_index,
        "target_output": target_output,
        "target_valid_mask": target_valid_mask,
        "active_demo_count": active_demo_count,
        "task_names": task_names,
        "query_indices": query_indices,
    }


def build_flow_context_dataloaders(
    args: argparse.Namespace,
    *,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[
    ARCContextFlowDataset,
    DataLoader,
    Optional[ARCContextFlowDataset],
    Optional[DataLoader],
    Optional[DistributedSampler],
    Optional[DistributedSampler],
]:
    root = Path(args.data_root)
    train_translation_aug = bool(getattr(args, "flow_train_translation_aug", False))
    train_resolution_aug = bool(getattr(args, "flow_train_resolution_aug", False))
    extra_specs: List[Tuple[Path, Optional[int], str, str]] = []
    if bool(getattr(args, "include_rearc", False)):
        rearc_limit = int(getattr(args, "rearc_limit", -1))
        extra_specs.append(
            (
                Path(getattr(args, "rearc_path", "raw_data/re_arc")),
                rearc_limit if rearc_limit >= 0 else None,
                "RE-ARC",
                "per_task",
            )
        )
    if bool(getattr(args, "include_barc", False)):
        barc_limit = int(getattr(args, "barc_limit", -1))
        extra_specs.append(
            (
                Path(getattr(args, "barc_path", "raw_data/BARC")),
                barc_limit if barc_limit >= 0 else None,
                "BARC",
                "total",
            )
        )
    train_dataset = ARCContextFlowDataset(
        root=root,
        split=args.train_split,
        mode="train",
        max_demos=args.max_demos,
        image_size=args.image_size,
        num_colors=args.num_colors,
        seed=args.seed,
        translation_augmentation=train_translation_aug,
        resolution_augmentation=train_resolution_aug,
        nested_dropout=bool(getattr(args, "nested_dropout", False)),
        extra_train_specs=extra_specs or None,
    )
    train_sampler: Optional[DistributedSampler] = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_flow_context,
        drop_last=False,
    )

    eval_dataset: Optional[ARCContextFlowDataset] = None
    eval_loader: Optional[DataLoader] = None
    eval_sampler: Optional[DistributedSampler] = None
    if args.eval_split:
        eval_dataset = ARCContextFlowDataset(
            root=root,
            split=args.eval_split,
            mode="eval",
            max_demos=args.max_demos,
            image_size=args.image_size,
            num_colors=args.num_colors,
            seed=args.seed,
            translation_augmentation=False,
            resolution_augmentation=False,
            nested_dropout=False,
        )
        if distributed:
            eval_sampler = DistributedSampler(
                eval_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            sampler=eval_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_flow_context,
            drop_last=False,
        )
    return train_dataset, train_loader, eval_dataset, eval_loader, train_sampler, eval_sampler
