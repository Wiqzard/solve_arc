from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


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
        num_demos: int,
        image_size: int,
        num_colors: int,
        seed: int = 42,
    ) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError("mode must be 'train' or 'eval'.")
        self.root = Path(root)
        self.split = split
        self.mode = mode
        self.num_demos = num_demos
        self.image_size = image_size
        self.num_colors = num_colors
        self.seed = seed

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
        return self._is_valid_grid(example["input"]) and self._is_valid_grid(example["output"])

    def _pad_grid(self, grid: Sequence[Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        canvas = torch.zeros((self.image_size, self.image_size), dtype=torch.long)
        mask = torch.zeros((self.image_size, self.image_size), dtype=torch.bool)
        array = torch.tensor(grid, dtype=torch.long)
        height, width = array.shape
        canvas[:height, :width] = array
        mask[:height, :width] = True
        return canvas, mask

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
            candidates = list(self._train_valid_indices[task_name])
        if len(candidates) >= self.num_demos:
            return rng.sample(candidates, k=self.num_demos)
        return [rng.choice(candidates) for _ in range(self.num_demos)]

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

        frames: List[torch.Tensor] = []
        frame_masks: List[torch.Tensor] = []

        for demo_idx in demo_indices:
            demo = train_examples[demo_idx]
            x_frame, x_mask = self._pad_grid(demo["input"])
            y_frame, y_mask = self._pad_grid(demo["output"])
            frames.extend([x_frame, y_frame])
            frame_masks.extend([x_mask, y_mask])

        query_input_frame, query_input_mask = self._pad_grid(query_example["input"])
        query_output_frame, query_output_mask = self._pad_grid(query_example["output"])
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
            "task_name": task_name,
            "query_index": torch.tensor(query_index, dtype=torch.long),
        }


def collate_flow_context(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    frames = torch.stack([item["frames"] for item in batch], dim=0)
    frame_valid_mask = torch.stack([item["frame_valid_mask"] for item in batch], dim=0)
    target_frame_index = torch.stack([item["target_frame_index"] for item in batch], dim=0)
    target_output = torch.stack([item["target_output"] for item in batch], dim=0)
    target_valid_mask = torch.stack([item["target_valid_mask"] for item in batch], dim=0)
    task_names = [item["task_name"] for item in batch]
    query_indices = torch.stack([item["query_index"] for item in batch], dim=0)

    return {
        "frames": frames,
        "frame_valid_mask": frame_valid_mask,
        "target_frame_index": target_frame_index,
        "target_output": target_output,
        "target_valid_mask": target_valid_mask,
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
    train_dataset = ARCContextFlowDataset(
        root=root,
        split=args.train_split,
        mode="train",
        num_demos=args.num_demos,
        image_size=args.image_size,
        num_colors=args.num_colors,
        seed=args.seed,
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
            num_demos=args.num_demos,
            image_size=args.image_size,
            num_colors=args.num_colors,
            seed=args.seed,
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
