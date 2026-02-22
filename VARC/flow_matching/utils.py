from __future__ import annotations

import torch
from torch import Tensor, nn


class ModelWrapper(nn.Module):
    """Base wrapper matching the API expected by flow-matching solvers."""

    def __init__(self, model: nn.Module | None = None) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        raise NotImplementedError


def categorical(probs: Tensor) -> Tensor:
    """Sample categorical values from probabilities over the last dimension."""
    if probs.dim() < 2:
        raise ValueError("categorical expects tensor with class dimension on the last axis.")
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    flat = probs.reshape(-1, probs.size(-1))
    sampled = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return sampled.view(*probs.shape[:-1])
