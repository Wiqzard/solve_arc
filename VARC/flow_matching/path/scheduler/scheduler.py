from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass
class SchedulerOutput:
    alpha_t: Tensor
    d_alpha_t: Tensor
    sigma_t: Tensor | None = None
    d_sigma_t: Tensor | None = None


class Scheduler(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, t: Tensor) -> SchedulerOutput:
        raise NotImplementedError


class ConvexScheduler(Scheduler):
    def __init__(self) -> None:
        super().__init__()

    def kappa(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def d_kappa(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, t: Tensor) -> SchedulerOutput:
        alpha_t = self.kappa(t)
        d_alpha_t = self.d_kappa(t)
        sigma_t = 1.0 - alpha_t
        d_sigma_t = -d_alpha_t
        return SchedulerOutput(alpha_t=alpha_t, d_alpha_t=d_alpha_t, sigma_t=sigma_t, d_sigma_t=d_sigma_t)


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


class VPScheduler(Scheduler):
    """
    Variance-preserving diffusion schedule.

    alpha_t = exp(-0.25 * (beta_max - beta_min) * t^2 - 0.5 * beta_min * t)
    sigma_t = sqrt(1 - alpha_t^2)
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        super().__init__()
        if beta_max < beta_min:
            raise ValueError("beta_max must be >= beta_min.")
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def forward(self, t: Tensor) -> SchedulerOutput:
        t = t.to(dtype=torch.float32)
        log_alpha = -0.25 * (self.beta_max - self.beta_min) * (t**2) - 0.5 * self.beta_min * t
        alpha_t = torch.exp(log_alpha)
        sigma_t = torch.sqrt((1.0 - alpha_t**2).clamp_min(1e-12))

        d_log_alpha = -0.5 * (self.beta_max - self.beta_min) * t - 0.5 * self.beta_min
        d_alpha_t = alpha_t * d_log_alpha
        d_sigma_t = -(alpha_t * d_alpha_t) / sigma_t.clamp_min(1e-12)
        return SchedulerOutput(alpha_t=alpha_t, d_alpha_t=d_alpha_t, sigma_t=sigma_t, d_sigma_t=d_sigma_t)


class LinearVPScheduler(Scheduler):
    """
    Linear VP schedule:
    alpha_t = t, sigma_t = sqrt(1 - t^2)
    """

    def forward(self, t: Tensor) -> SchedulerOutput:
        t = t.to(dtype=torch.float32)
        alpha_t = t
        sigma_t = torch.sqrt((1.0 - t**2).clamp_min(1e-12))
        d_alpha_t = torch.ones_like(t)
        d_sigma_t = -t / sigma_t.clamp_min(1e-12)
        return SchedulerOutput(alpha_t=alpha_t, d_alpha_t=d_alpha_t, sigma_t=sigma_t, d_sigma_t=d_sigma_t)


class CosineScheduler(Scheduler):
    """
    Cosine schedule:
    alpha_t = sin(pi * t / 2), sigma_t = cos(pi * t / 2)
    """

    def forward(self, t: Tensor) -> SchedulerOutput:
        t = t.to(dtype=torch.float32)
        angle = 0.5 * math.pi * t
        alpha_t = torch.sin(angle)
        sigma_t = torch.cos(angle)
        d_alpha_t = 0.5 * math.pi * torch.cos(angle)
        d_sigma_t = -0.5 * math.pi * torch.sin(angle)
        return SchedulerOutput(alpha_t=alpha_t, d_alpha_t=d_alpha_t, sigma_t=sigma_t, d_sigma_t=d_sigma_t)
