import torch
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
import json
import random
import argparse
import random
import numpy as np

IGNORE_INDEX = 10
PAD_INDEX = 11
MAX_SIZE = 30

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    For parallel training
    """
    inputs = torch.stack([item["inputs"] for item in batch], dim=0)
    attention = torch.stack([item["attention_mask"] for item in batch], dim=0)
    targets = torch.stack([item["targets"] for item in batch], dim=0)
    task_ids = torch.stack([item["task_id"] for item in batch], dim=0)
    target_shapes = torch.stack([item["target_shape"] for item in batch], dim=0)
    example_indices = torch.stack([item["example_index"] for item in batch], dim=0)
    offset = torch.stack([torch.tensor(item["offset"]) for item in batch], dim=0)
    scale_factors = torch.stack([torch.tensor(item["scale_factor"]) for item in batch], dim=0)
    task_names = [item["task_name"] for item in batch]
    raw_inputs = [item["raw_input"] for item in batch]
    raw_outputs = [item["raw_output"] for item in batch]
    return {
        "inputs": inputs,
        "attention_mask": attention,
        "targets": targets,
        "task_ids": task_ids,
        "target_shapes": target_shapes,
        "example_indices": example_indices,
        "task_names": task_names,
        "raw_inputs": raw_inputs,
        "raw_outputs": raw_outputs,
        "offset": offset,
        "scale_factors": scale_factors,
    }


def pad_grid_with_translation(grid: List[List[int]], max_size: int, x_offset: int, y_offset: int, output_shape=True) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Do random translation and padding
    Returns:
        padded_tensor: (max_size, max_size) tensor with padding
        mask: (max_size, max_size) tensor, 1 for valid positions, 0 for padding
        height: height of original grid
        width: width of original grid
    """
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    if height > max_size - 2 or width > max_size - 2:
        raise ValueError(
            f"Grid size ({height}, {width}) exceeds configured max_size={max_size - 2}."
        )

    tensor = torch.full((max_size, max_size), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros((max_size, max_size), dtype=torch.long)

    values = torch.tensor(grid, dtype=torch.long)
    tensor[y_offset:y_offset+height, x_offset:x_offset+width] = values
    mask[y_offset:y_offset+height, x_offset:x_offset+width] = 1

    if output_shape:
        tensor[y_offset: y_offset + height, x_offset + width] = PAD_INDEX
        tensor[y_offset + height, x_offset: x_offset + width + 1] = PAD_INDEX
        mask[y_offset:y_offset+height+1, x_offset:x_offset+width+1] = 1
    return tensor, mask, height, width

def resolution_augmentation(example, max_cur_x, max_cur_y, rng, img_size=60):
    """
    Do resolution augmentation by random scaling
    1. Randomly choose a scale factor
    2. Scale up the input and output grids
    3. Return the new example and scale factor
    """
    max_len = max(max_cur_x, max_cur_y)
    max_scale_factor = (img_size // max_len)
    scale_factor = rng.randint(1, max_scale_factor)
    new_example = {}
    new_example['input'] = np.repeat(np.repeat(example['input'], scale_factor, axis=0), scale_factor, axis=1).tolist()
    new_example['output'] = np.repeat(np.repeat(example['output'], scale_factor, axis=0), scale_factor, axis=1).tolist()

    return new_example, scale_factor

class ARCDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        subset: str = "train",
        max_size: int = 32,
        task_lookup: Optional[Dict[str, int]] = None,
        extra_train_roots: Optional[Iterable[Path]] = None,
        extra_train_limit: Optional[int] = None,
        extra_train_specs: Optional[Iterable[Tuple[Path, Optional[int], str]]] = None,
    ) -> None:
        if subset not in {"train", "test"}:
            raise ValueError("subset must be 'train' or 'test'.")
        self.rng = random.Random(42)
        self.root = Path(root)
        self.max_size = max_size
        self.subset = subset

        self.samples: List[Dict[str, torch.Tensor]] = []
        self.task_lookup: Dict[str, int] = dict(task_lookup) if task_lookup is not None else {}
        self._next_example_index_by_task: Dict[str, int] = {}

        split_dir = self.root / "data" / split
        files = sorted(split_dir.glob("*.json"))
        examples_key = "train" if subset == "train" else "test"
        self.translation_enabled = True
        self.fix_scale_factor = 2
        self.resolution_enabled = True

        print('Start loading data...')
        for file_path in files:
            task_name = file_path.stem
            if task_lookup is None:
                if task_name in self.task_lookup:
                    task_index = self.task_lookup[task_name]
                else:
                    task_index = len(self.task_lookup)
                    self.task_lookup[task_name] = task_index
            else:
                if task_name not in self.task_lookup:
                    continue
                task_index = self.task_lookup[task_name]

            with file_path.open("r") as fh:
                task_data = json.load(fh)

            examples = task_data.get(examples_key, [])
         
            # Load the original training/test examples, and do translation and scaling augmentation on-the-fly
            for example_index, example in enumerate(examples):
                max_cur_y = len(example["input"])
                max_cur_x = len(example["input"][0])
                if "output" in example:
                    max_cur_y = max(max_cur_y, len(example["output"]))
                    max_cur_x = max(max_cur_x, len(example["output"][0]))
                if max_cur_y > MAX_SIZE or max_cur_x > MAX_SIZE:
                    continue

                self.samples.append(
                    {"example": example, "task_index": task_index,
                        "task_name": task_name, "example_index": example_index}
                )
                self._next_example_index_by_task[task_name] = example_index + 1
        # Load RE-ARC extra training data if specified
        if subset == "train":
            if extra_train_roots:
                for extra_root in extra_train_roots:
                    self._load_extra_training_data(
                        extra_root=Path(extra_root),
                        limit_per_task=extra_train_limit,
                        source_name="extra",
                    )
            if extra_train_specs:
                for extra_root, per_task_limit, source_name in extra_train_specs:
                    self._load_extra_training_data(
                        extra_root=Path(extra_root),
                        limit_per_task=per_task_limit,
                        source_name=source_name,
                    )

        if not self.samples:
            raise RuntimeError(
                f"No samples found for split '{split}' subset '{subset}' under {split_dir}."
            )
        print('Finish loading!!')
        self.num_tasks = len(self.task_lookup)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cur_batch = self.samples[idx]
        processed_random_translation_batch = self.process_per_example(
            example=cur_batch["example"],
            task_index=cur_batch["task_index"],
            task_name=cur_batch["task_name"],
            example_index=cur_batch["example_index"],
            rng=self.rng,
            if_translation=self.translation_enabled,
        )
        return processed_random_translation_batch

    def enable_translation(self):
        self.translation_enabled = True

    def disable_resolution_augmentation(self, fix_scale_factor=2):
        self.fix_scale_factor = fix_scale_factor
        self.resolution_enabled = False
    
    def enable_resolution_augmentation(self):
        self.resolution_enabled = True

    def disable_translation(self):
        self.translation_enabled = False
    
    def _get_or_add_task_index(self, task_name: str) -> int:
        if task_name in self.task_lookup:
            return self.task_lookup[task_name]
        task_index = len(self.task_lookup)
        self.task_lookup[task_name] = task_index
        return task_index

    def _is_valid_example(self, example: Any) -> bool:
        if not isinstance(example, dict):
            return False
        if "input" not in example or "output" not in example:
            return False
        if not isinstance(example["input"], list) or not example["input"]:
            return False
        if not isinstance(example["output"], list) or not example["output"]:
            return False
        max_cur_y = len(example["input"])
        max_cur_x = len(example["input"][0]) if max_cur_y > 0 else 0
        max_cur_y = max(max_cur_y, len(example["output"]))
        max_cur_x = max(max_cur_x, len(example["output"][0]) if len(example["output"]) > 0 else 0)
        return max_cur_y <= MAX_SIZE and max_cur_x <= MAX_SIZE

    def process_per_example(self, example, task_index, task_name, example_index, rng, if_translation=True):
        """
        Conduct random translation and resolution augmentation for each example
        1. Randomly scale up the input and output grids
        2. Randomly translate the grids within the max_size
        3. Pad the grids to max_size
        4. Return the processed tensors and other info
        Returns:
            A dictionary containing:
                inputs: (max_size, max_size) tensor of input grid with padding
                attention_mask: (max_size, max_size) tensor, 1 for valid positions, 0 for padding
                targets: (max_size, max_size) tensor of output grid with padding
                task_id: tensor of task index
                task_name: task name string
                example_index: tensor of example index
                target_shape: tensor of (height, width) of original output grid
                raw_input: original input grid
                raw_output: original output grid
                offset: (x_offset, y_offset) used for translation
                scale_factor: scale factor used for resolution augmentation
        """
        max_cur_y = len(example["input"])
        max_cur_x = len(example["input"][0]) 
        if "output" in example:
            max_cur_y = max(max_cur_y, len(example["output"]))
            max_cur_x = max(max_cur_x, len(example["output"][0]))
        max_img_size = self.max_size - 2  # Leave border
        max_size = self.max_size
        if self.resolution_enabled:
            example, scale_factor = resolution_augmentation(example, max_cur_x, max_cur_y, rng, img_size=max_img_size)
        else:
            scale_factor = self.fix_scale_factor
            new_example = {}
            new_example['input'] = np.repeat(np.repeat(example['input'], scale_factor, axis=0), scale_factor, axis=1).tolist()
            new_example['output'] = np.repeat(np.repeat(example['output'], scale_factor, axis=0), scale_factor, axis=1).tolist()
            example = new_example
            
        max_cur_x = max_cur_x * scale_factor
        max_cur_y = max_cur_y * scale_factor

        if if_translation:
            x_offset = rng.randint(1, max_img_size - max_cur_x) if max_img_size > max_cur_x else 1
            y_offset = rng.randint(1, max_img_size - max_cur_y) if max_img_size > max_cur_y else 1
        else:
            x_offset = 1
            y_offset = 1

        input_grid, input_mask, _, _ = pad_grid_with_translation(example["input"], max_size, x_offset, y_offset, output_shape=False)

        if "output" in example:
            target_grid, target_mask, target_h, target_w = pad_grid_with_translation(example["output"], max_size, x_offset, y_offset, output_shape=True)
        else:
            target_grid = torch.full((max_size, max_size), IGNORE_INDEX, dtype=torch.long)
            target_mask = torch.zeros((max_size, max_size), dtype=torch.long)
            target_h = 0
            target_w = 0

        target_grid = target_grid.clone()
        target_grid[target_mask == 0] = IGNORE_INDEX

        raw_input = example.get("input", [])
        raw_output = example.get("output") if "output" in example else None

        return {
                "inputs": input_grid,
                "attention_mask": input_mask,
                "targets": target_grid,
                "task_id": torch.tensor(task_index, dtype=torch.long),
                "task_name": task_name,
                "example_index": torch.tensor(example_index, dtype=torch.long),
                "target_shape": torch.tensor([target_h, target_w], dtype=torch.long),
                "raw_input": raw_input,
                "raw_output": raw_output,
                "offset": (x_offset, y_offset),
                "scale_factor": scale_factor,
            }


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

        if isinstance(payload.get("train"), list):
            task_sets.append((default_task_name, payload["train"]))
            return task_sets

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
    ) -> None:
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
        added_samples = 0
        added_per_task: Dict[str, int] = {}
        for file_path in files:
            with file_path.open("r") as fh:
                payload = json.load(fh)
            task_sets = self._iter_task_examples_from_payload(
                payload,
                default_task_name=file_path.stem,
            )
            for task_name, examples in task_sets:
                valid_examples = [ex for ex in examples if self._is_valid_example(ex)]
                if not valid_examples:
                    continue

                if limit_per_task is not None:
                    rng.shuffle(valid_examples)
                    remaining = limit_per_task - added_per_task.get(task_name, 0)
                    if remaining <= 0:
                        continue
                    valid_examples = valid_examples[:remaining]

                task_index = self._get_or_add_task_index(task_name)
                next_example_index = self._next_example_index_by_task.get(task_name, 0)
                for offset, example in enumerate(valid_examples):
                    self.samples.append(
                        {
                            "example": example,
                            "task_index": task_index,
                            "task_name": task_name,
                            "example_index": next_example_index + offset,
                        }
                    )
                self._next_example_index_by_task[task_name] = next_example_index + len(valid_examples)
                added_per_task[task_name] = added_per_task.get(task_name, 0) + len(valid_examples)
                added_samples += len(valid_examples)

        print(f"Added {added_samples} {source_name} samples from {source_location}.")
        return None

    
def build_dataloaders(
    args: argparse.Namespace,
    *,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    root = Path(args.data_root)
    extra_specs: List[Tuple[Path, Optional[int], str]] = []
    if getattr(args, "include_rearc", False):
        rearc_limit = args.rearc_limit if args.rearc_limit >= 0 else None
        extra_specs.append((Path(args.rearc_path), rearc_limit, "RE-ARC"))
    if getattr(args, "include_barc", False):
        barc_limit = args.barc_limit if args.barc_limit >= 0 else None
        extra_specs.append((Path(args.barc_path), barc_limit, "BARC"))


    train_split = getattr(args, "train_split", "training")
    train_dataset = ARCDataset(
        root,
        train_split,
        subset="train",
        max_size=args.image_size,
        extra_train_specs=extra_specs or None,
    )
    train_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
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
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    eval_loader = None
    eval_sampler = None
    if args.eval_split:
        eval_subset = getattr(args, "eval_subset", "test")
        eval_dataset = ARCDataset(
            root,
            args.eval_split,
            subset=eval_subset,
            max_size=args.image_size,
            task_lookup=train_dataset.task_lookup,
        )
        if distributed:
            eval_sampler = torch.utils.data.distributed.DistributedSampler(
                eval_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=eval_sampler,
            drop_last=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
    else:
        eval_dataset = None

    return train_dataset, train_loader, eval_dataset, eval_loader, train_sampler, eval_sampler
