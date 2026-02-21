from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional


TASK_ID_KEYS = (
    "task_id",
    "task_name",
    "problem_id",
    "arc_task_id",
    "id",
    "key",
    "name",
)
EXAMPLE_LIST_KEYS = ("train", "examples", "pairs", "samples", "records", "data", "items")
TASK_CONTAINER_KEYS = ("tasks", "task_map", "problems")
INPUT_KEYS = ("input", "x", "in", "input_grid", "input_image")
OUTPUT_KEYS = ("output", "y", "out", "output_grid", "output_image")
SERIALIZED_KEYS = ("task", "problem", "arc_task", "json", "task_json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare BARC into VARC extra-training format: output_root/tasks/<task>.json",
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default=None,
        help="Local BARC path (file or directory with .json/.jsonl files).",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="Optional Hugging Face dataset ID to download and convert (for example: org/name).",
    )
    parser.add_argument("--hf-config", type=str, default=None, help="Optional Hugging Face dataset config.")
    parser.add_argument("--hf-split", type=str, default="train", help="Hugging Face split to read.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="raw_data/BARC",
        help="Output root. Converted files are written to output_root/tasks/*.json.",
    )
    parser.add_argument(
        "--max-examples-per-task",
        type=int,
        default=-1,
        help="Cap examples per task after conversion (-1 means all).",
    )
    parser.add_argument(
        "--min-examples-per-task",
        type=int,
        default=1,
        help="Drop tasks with fewer than this many valid examples.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output_root/tasks if it already exists.",
    )
    return parser.parse_args()


def _normalize_task_name(name: str, fallback: str) -> str:
    text = str(name).strip()
    if not text:
        text = fallback
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return safe or fallback


def _get_task_name(obj: Dict[str, Any], fallback: str) -> str:
    for key in TASK_ID_KEYS:
        if key in obj and obj[key] is not None:
            return _normalize_task_name(str(obj[key]), fallback)
    return fallback


def _normalize_grid(grid: Any) -> Optional[List[List[int]]]:
    if not isinstance(grid, list) or len(grid) == 0:
        return None
    rows: List[List[int]] = []
    width: Optional[int] = None
    for row in grid:
        if not isinstance(row, list) or len(row) == 0:
            return None
        converted: List[int] = []
        for value in row:
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                converted.append(int(value))
                continue
            if isinstance(value, float) and value.is_integer():
                converted.append(int(value))
                continue
            return None
        if width is None:
            width = len(converted)
        elif len(converted) != width:
            return None
        rows.append(converted)
    return rows


def _dict_get_any(obj: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        if key in obj:
            return obj[key]
    return None


def _parse_if_json_string(node: Any) -> Optional[Any]:
    if not isinstance(node, str):
        return None
    text = node.strip()
    if not text or text[0] not in "[{":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_examples(node: Any) -> List[Dict[str, List[List[int]]]]:
    parsed = _parse_if_json_string(node)
    if parsed is not None:
        return _extract_examples(parsed)

    if isinstance(node, tuple):
        node = list(node)

    examples: List[Dict[str, List[List[int]]]] = []
    if isinstance(node, dict):
        input_candidate = _dict_get_any(node, INPUT_KEYS)
        output_candidate = _dict_get_any(node, OUTPUT_KEYS)
        if input_candidate is not None and output_candidate is not None:
            input_grid = _normalize_grid(input_candidate)
            output_grid = _normalize_grid(output_candidate)
            if input_grid is not None and output_grid is not None:
                return [{"input": input_grid, "output": output_grid}]

        for key in EXAMPLE_LIST_KEYS:
            value = node.get(key)
            if isinstance(value, (list, dict, tuple, str)):
                examples.extend(_extract_examples(value))

        for key in SERIALIZED_KEYS:
            value = node.get(key)
            if isinstance(value, (list, dict, tuple, str)):
                examples.extend(_extract_examples(value))
        return examples

    if isinstance(node, list):
        # Pair format: [input_grid, output_grid]
        if len(node) == 2:
            input_grid = _normalize_grid(node[0])
            output_grid = _normalize_grid(node[1])
            if input_grid is not None and output_grid is not None:
                return [{"input": input_grid, "output": output_grid}]
        for item in node:
            examples.extend(_extract_examples(item))
    return examples


def _collect_task_examples(
    payload: Any,
    *,
    source_name: str,
    task_examples: DefaultDict[str, List[Dict[str, List[List[int]]]]],
) -> None:
    if isinstance(payload, dict):
        for container_key in TASK_CONTAINER_KEYS:
            container = payload.get(container_key)
            if isinstance(container, dict):
                for raw_task_name, task_payload in container.items():
                    task_name = _normalize_task_name(str(raw_task_name), source_name)
                    task_examples[task_name].extend(_extract_examples(task_payload))
                return

        if isinstance(payload.get("train"), list):
            task_name = _get_task_name(payload, source_name)
            task_examples[task_name].extend(_extract_examples(payload["train"]))
            return

        direct_examples = _extract_examples(payload)
        if direct_examples:
            task_name = _get_task_name(payload, source_name)
            task_examples[task_name].extend(direct_examples)
            return

        # Flat task map format: {task_id: {...}} or {task_id: [...]}
        if _dict_get_any(payload, INPUT_KEYS) is None and _dict_get_any(payload, OUTPUT_KEYS) is None:
            found_task_map = False
            for raw_task_name, task_payload in payload.items():
                if isinstance(task_payload, (dict, list)):
                    current = _extract_examples(task_payload)
                    if current:
                        task_name = _normalize_task_name(str(raw_task_name), source_name)
                        task_examples[task_name].extend(current)
                        found_task_map = True
            if found_task_map:
                return

        task_name = _get_task_name(payload, source_name)
        task_examples[task_name].extend(direct_examples)
        return

    if isinstance(payload, list):
        has_per_item_task_id = any(
            isinstance(item, dict) and any(task_key in item for task_key in TASK_ID_KEYS) for item in payload
        )
        if has_per_item_task_id:
            for idx, item in enumerate(payload):
                if not isinstance(item, dict):
                    continue
                task_name = _get_task_name(item, f"{source_name}_{idx:06d}")
                task_examples[task_name].extend(_extract_examples(item))
            return

        task_examples[source_name].extend(_extract_examples(payload))


def _collect_from_json_file(
    file_path: Path,
    task_examples: DefaultDict[str, List[Dict[str, List[List[int]]]]],
) -> None:
    source_name = _normalize_task_name(file_path.stem, "task")
    if file_path.suffix.lower() == ".jsonl":
        with file_path.open("r", encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                _collect_task_examples(
                    payload,
                    source_name=f"{source_name}_{line_idx:06d}",
                    task_examples=task_examples,
                )
        return

    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    _collect_task_examples(payload, source_name=source_name, task_examples=task_examples)


def _collect_from_local(
    input_root: Path,
    task_examples: DefaultDict[str, List[Dict[str, List[List[int]]]]],
) -> None:
    if input_root.is_file():
        if input_root.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError(f"Unsupported file suffix for {input_root}; expected .json or .jsonl.")
        _collect_from_json_file(input_root, task_examples)
        return

    if not input_root.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_root}")
    if not input_root.is_dir():
        raise ValueError(f"Input path must be a directory or json/jsonl file: {input_root}")

    files = sorted(
        [path for path in input_root.rglob("*.json")] + [path for path in input_root.rglob("*.jsonl")]
    )
    if not files:
        raise RuntimeError(f"No .json/.jsonl files found under {input_root}")
    for file_path in files:
        _collect_from_json_file(file_path, task_examples)


def _collect_from_hf(
    *,
    dataset_id: str,
    config: Optional[str],
    split: str,
    task_examples: DefaultDict[str, List[Dict[str, List[List[int]]]]],
) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("`datasets` is required for --hf-dataset. Install dependencies first.") from exc

    dataset = load_dataset(dataset_id, name=config, split=split)
    for row_idx, row in enumerate(dataset):
        source_name = _normalize_task_name(f"{dataset_id}_{split}_{row_idx:08d}", "hf_task")
        _collect_task_examples(row, source_name=source_name, task_examples=task_examples)


def _dedupe_examples(examples: Iterable[Dict[str, List[List[int]]]]) -> List[Dict[str, List[List[int]]]]:
    unique: List[Dict[str, List[List[int]]]] = []
    seen = set()
    for example in examples:
        key = json.dumps(example, sort_keys=True, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(example)
    return unique


def main() -> None:
    args = parse_args()
    if args.input_root is None and args.hf_dataset is None:
        raise ValueError("Pass at least one source: --input-root and/or --hf-dataset.")

    rng = random.Random(args.seed)
    task_examples: DefaultDict[str, List[Dict[str, List[List[int]]]]] = defaultdict(list)

    if args.input_root is not None:
        _collect_from_local(Path(args.input_root), task_examples)
    if args.hf_dataset is not None:
        _collect_from_hf(
            dataset_id=args.hf_dataset,
            config=args.hf_config,
            split=args.hf_split,
            task_examples=task_examples,
        )

    output_root = Path(args.output_root)
    tasks_dir = output_root / "tasks"
    if tasks_dir.exists():
        if args.overwrite:
            shutil.rmtree(tasks_dir)
        else:
            raise FileExistsError(f"{tasks_dir} already exists. Pass --overwrite to replace it.")
    tasks_dir.mkdir(parents=True, exist_ok=True)

    max_per_task = args.max_examples_per_task if args.max_examples_per_task >= 0 else None
    min_per_task = max(args.min_examples_per_task, 1)

    written_tasks = 0
    written_examples = 0
    for task_name in sorted(task_examples.keys()):
        deduped = _dedupe_examples(task_examples[task_name])
        if max_per_task is not None and len(deduped) > max_per_task:
            rng.shuffle(deduped)
            deduped = deduped[:max_per_task]
        if len(deduped) < min_per_task:
            continue

        file_path = tasks_dir / f"{task_name}.json"
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(deduped, handle, separators=(",", ":"))
        written_tasks += 1
        written_examples += len(deduped)

    if written_tasks == 0:
        raise RuntimeError(
            "No valid task files were produced. Check your source format and conversion arguments."
        )

    print(
        f"Wrote {written_tasks} task files and {written_examples} examples to {tasks_dir}."
    )
    print("You can now train with: --include-barc --barc-path " + str(output_root))


if __name__ == "__main__":
    main()
