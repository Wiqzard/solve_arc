from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor

ARC_COLORS = {
    0: (0, 0, 0),
    1: (0, 116, 217),
    2: (255, 65, 54),
    3: (46, 204, 64),
    4: (255, 220, 0),
    5: (170, 170, 170),
    6: (240, 18, 190),
    7: (255, 133, 27),
    8: (127, 219, 255),
    9: (177, 13, 201),
    10: (230, 230, 230),
    11: (255, 255, 255),
}

DEFAULT_CELL_SIZE = 18
DEFAULT_PANEL_PADDING = 10
DEFAULT_MAX_CONTEXT = 4


def base_task_name(task_name: str) -> str:
    if "_" not in task_name:
        return task_name
    return task_name.split("_", 1)[0]


def _to_2d_list(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    if isinstance(grid, np.ndarray):
        return grid.tolist()
    return [list(row) for row in grid]


def render_arc_grid(
    grid: Sequence[Sequence[int]],
    *,
    cell_size: int = DEFAULT_CELL_SIZE,
) -> Image.Image:
    grid_list = _to_2d_list(grid)
    if not grid_list:
        return Image.new("RGB", (cell_size, cell_size), color=(255, 255, 255))

    height = len(grid_list)
    width = len(grid_list[0]) if height > 0 else 0
    image = Image.new("RGB", (max(1, width) * cell_size, max(1, height) * cell_size), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    for y, row in enumerate(grid_list):
        for x, value in enumerate(row):
            color = ARC_COLORS.get(int(value), (0, 0, 0))
            x0 = x * cell_size
            y0 = y * cell_size
            draw.rectangle((x0, y0, x0 + cell_size, y0 + cell_size), fill=color)
            draw.rectangle((x0, y0, x0 + cell_size, y0 + cell_size), outline=(70, 70, 70), width=1)
    return image


def compose_task_image(
    train_examples: Sequence[Dict[str, Any]],
    query_input: Sequence[Sequence[int]],
    candidate_output: Sequence[Sequence[int]],
    *,
    max_context: int = DEFAULT_MAX_CONTEXT,
) -> Image.Image:
    panels: List[Image.Image] = []
    for idx, pair in enumerate(train_examples[:max_context]):
        input_img = render_arc_grid(pair["input"])
        output_img = render_arc_grid(pair["output"])
        gap = Image.new("RGB", (DEFAULT_PANEL_PADDING, max(input_img.height, output_img.height)), color=(255, 255, 255))
        panel = Image.new(
            "RGB",
            (input_img.width + output_img.width + gap.width + 2 * DEFAULT_PANEL_PADDING, max(input_img.height, output_img.height) + 3 * DEFAULT_PANEL_PADDING),
            color=(255, 255, 255),
        )
        panel.paste(input_img, (DEFAULT_PANEL_PADDING, DEFAULT_PANEL_PADDING))
        panel.paste(gap, (DEFAULT_PANEL_PADDING + input_img.width, DEFAULT_PANEL_PADDING))
        panel.paste(output_img, (DEFAULT_PANEL_PADDING + input_img.width + gap.width, DEFAULT_PANEL_PADDING))
        draw = ImageDraw.Draw(panel)
        draw.text((DEFAULT_PANEL_PADDING, panel.height - DEFAULT_PANEL_PADDING * 2), f"demo {idx + 1}", fill=(0, 0, 0))
        panels.append(panel)

    q_img = render_arc_grid(query_input)
    c_img = render_arc_grid(candidate_output)
    query_panel = Image.new(
        "RGB",
        (q_img.width + c_img.width + DEFAULT_PANEL_PADDING + 2 * DEFAULT_PANEL_PADDING, max(q_img.height, c_img.height) + 3 * DEFAULT_PANEL_PADDING),
        color=(255, 255, 255),
    )
    query_panel.paste(q_img, (DEFAULT_PANEL_PADDING, DEFAULT_PANEL_PADDING))
    query_panel.paste(c_img, (DEFAULT_PANEL_PADDING + q_img.width + DEFAULT_PANEL_PADDING, DEFAULT_PANEL_PADDING))
    draw = ImageDraw.Draw(query_panel)
    draw.text((DEFAULT_PANEL_PADDING, query_panel.height - DEFAULT_PANEL_PADDING * 2), "query + proposal", fill=(0, 0, 0))
    panels.append(query_panel)

    total_width = max(panel.width for panel in panels)
    total_height = sum(panel.height for panel in panels) + DEFAULT_PANEL_PADDING * (len(panels) + 1)
    stitched = Image.new("RGB", (total_width + 2 * DEFAULT_PANEL_PADDING, total_height), color=(255, 255, 255))
    y = DEFAULT_PANEL_PADDING
    for panel in panels:
        stitched.paste(panel, (DEFAULT_PANEL_PADDING, y))
        y += panel.height + DEFAULT_PANEL_PADDING
    return stitched


def build_reward_prompt(
    train_examples: Sequence[Dict[str, Any]],
    query_input: Sequence[Sequence[int]],
    candidate_output: Sequence[Sequence[int]],
    *,
    max_context: int = DEFAULT_MAX_CONTEXT,
) -> str:
    truncated_context = list(train_examples[:max_context])
    payload = {
        "demonstrations": truncated_context,
        "query_input": query_input,
        "candidate_output": candidate_output,
        "instruction": (
            "Decide whether candidate_output correctly solves the ARC query_input "
            "under the transformation shown in demonstrations. "
            "Respond with exactly one token: yes or no."
        ),
        "output_format": {"answer": "yes|no"},
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


class TaskContextCache:
    def __init__(self, data_root: Path, split: str) -> None:
        self.data_root = data_root
        self.split = split
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _load_task(self, task_name: str) -> Dict[str, Any]:
        candidate_names = [task_name]
        base_name = base_task_name(task_name)
        if base_name != task_name:
            candidate_names.append(base_name)

        last_path: Optional[Path] = None
        for name in candidate_names:
            task_path = self.data_root / "data" / self.split / f"{name}.json"
            last_path = task_path
            if task_path.exists():
                with task_path.open("r") as handle:
                    return json.load(handle)
        raise FileNotFoundError(f"Unable to find ARC task JSON for '{task_name}' under {last_path}.")

    def get_train_examples(self, task_name: str) -> List[Dict[str, Any]]:
        if task_name not in self._cache:
            self._cache[task_name] = self._load_task(task_name)
        payload = self._cache[task_name]
        return payload.get("train", [])


def _resolve_model_class():
    import transformers

    for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
        model_cls = getattr(transformers, class_name, None)
        if model_cls is not None:
            return model_cls
    raise RuntimeError("Unable to resolve a compatible transformers AutoModel class for VLM reward scoring.")


def _collect_single_token_ids(tokenizer, candidates: Sequence[str], label: str) -> List[int]:
    ids: List[int] = []
    for candidate in candidates:
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(token_ids) == 1:
            token_id = int(token_ids[0])
            if token_id not in ids:
                ids.append(token_id)
    if not ids:
        raise ValueError(f"Could not find a single-token representation for '{label}'.")
    return ids


@dataclass
class YesNoReward:
    reward: float
    yes_logit: float
    no_logit: float


class QwenYesNoRewardModel:
    def __init__(
        self,
        *,
        model_id: str,
        device: torch.device,
        dtype: str = "bfloat16",
        use_image: bool = True,
        max_context_examples: int = DEFAULT_MAX_CONTEXT,
    ) -> None:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported reward dtype '{dtype}'. Supported values: {sorted(dtype_map.keys())}")
        self.device = device
        self.use_image = use_image
        self.max_context_examples = max_context_examples

        model_cls = _resolve_model_class()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = model_cls.from_pretrained(
            model_id,
            torch_dtype=dtype_map[dtype],
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Reward model processor does not expose a tokenizer; cannot compute yes/no logits.")
        self.tokenizer = tokenizer
        self.yes_token_ids = _collect_single_token_ids(
            self.tokenizer,
            (" yes", "yes", " Yes", "Yes", "\nyes", "\nYes"),
            "yes",
        )
        self.no_token_ids = _collect_single_token_ids(
            self.tokenizer,
            (" no", "no", " No", "No", "\nno", "\nNo"),
            "no",
        )

    def _build_inputs(self, prompt: str, image: Optional[Image.Image]) -> Dict[str, torch.Tensor]:
        if image is not None and self.use_image:
            if hasattr(self.processor, "apply_chat_template"):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                formatted_prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                model_inputs = self.processor(text=[formatted_prompt], images=[image], return_tensors="pt")
            else:
                model_inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        else:
            if hasattr(self.processor, "apply_chat_template"):
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ]
                formatted_prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                model_inputs = self.processor(text=[formatted_prompt], return_tensors="pt")
            else:
                model_inputs = self.processor(text=[prompt], return_tensors="pt")

        return {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in model_inputs.items()}

    @torch.no_grad()
    def score_candidate(
        self,
        *,
        train_examples: Sequence[Dict[str, Any]],
        query_input: Sequence[Sequence[int]],
        candidate_output: Sequence[Sequence[int]],
    ) -> YesNoReward:
        prompt = build_reward_prompt(
            train_examples=train_examples,
            query_input=query_input,
            candidate_output=candidate_output,
            max_context=self.max_context_examples,
        )
        image = None
        if self.use_image:
            image = compose_task_image(
                train_examples=train_examples,
                query_input=query_input,
                candidate_output=candidate_output,
                max_context=self.max_context_examples,
            )
        model_inputs = self._build_inputs(prompt, image)
        outputs = self.model(**model_inputs)
        next_token_logits = outputs.logits[:, -1, :]
        yes_slice = next_token_logits[0, self.yes_token_ids]
        no_slice = next_token_logits[0, self.no_token_ids]
        yes_logit = float(torch.logsumexp(yes_slice, dim=0).detach().cpu())
        no_logit = float(torch.logsumexp(no_slice, dim=0).detach().cpu())
        return YesNoReward(reward=yes_logit - no_logit, yes_logit=yes_logit, no_logit=no_logit)
