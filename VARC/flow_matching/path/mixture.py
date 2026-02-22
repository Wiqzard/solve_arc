from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .scheduler.scheduler import CondOTScheduler, ConvexScheduler, SchedulerOutput


@dataclass
class DiscretePathSample:
    x_t: Tensor
    scheduler_output: SchedulerOutput


class MixtureDiscreteProbPath:
    """Discrete mixture probability path used by x-prediction training."""

    def __init__(self, scheduler: ConvexScheduler | None = None) -> None:
        self.scheduler = scheduler if scheduler is not None else CondOTScheduler()

    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> DiscretePathSample:
        if x_0.shape != x_1.shape:
            raise ValueError("x_0 and x_1 must have identical shape.")
        scheduler_output = self.scheduler(t)
        alpha_t = scheduler_output.alpha_t
        while alpha_t.dim() < x_1.dim():
            alpha_t = alpha_t.unsqueeze(-1)

        mask = torch.rand_like(x_1.float()) < alpha_t
        x_t = torch.where(mask, x_1, x_0)
        return DiscretePathSample(x_t=x_t, scheduler_output=scheduler_output)
