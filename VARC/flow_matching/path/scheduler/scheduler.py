from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class SchedulerOutput:
    alpha_t: Tensor
    d_alpha_t: Tensor


class ConvexScheduler(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def kappa(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def d_kappa(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(alpha_t=self.kappa(t), d_alpha_t=self.d_kappa(t))


class CondOTScheduler(ConvexScheduler):
    """Default scheduler used in the flow-matching reference package."""

    def kappa(self, t: Tensor) -> Tensor:
        return t

    def d_kappa(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)


class PolynomialConvexScheduler(ConvexScheduler):
    def __init__(self, n: float = 2.0) -> None:
        super().__init__()
        self.n = float(n)

    def kappa(self, t: Tensor) -> Tensor:
        return t ** self.n

    def d_kappa(self, t: Tensor) -> Tensor:
        if self.n == 0:
            return torch.zeros_like(t)
        return self.n * (t ** (self.n - 1.0))


class ExponentialScheduler(ConvexScheduler):
    """
    ARC-compatible exponential scheduler:
    alpha_t = 1 - exp(-beta * t), d_alpha_t = beta * exp(-beta * t).
    """

    def __init__(self, beta: float = 5.0) -> None:
        super().__init__()
        self.beta = float(beta)

    def kappa(self, t: Tensor) -> Tensor:
        return 1.0 - torch.exp(-self.beta * t)

    def d_kappa(self, t: Tensor) -> Tensor:
        return self.beta * torch.exp(-self.beta * t)
