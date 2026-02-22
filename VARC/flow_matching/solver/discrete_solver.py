from __future__ import annotations

from contextlib import nullcontext
from math import ceil
from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.solver.solver import Solver
from flow_matching.utils import ModelWrapper, categorical
from .utils import get_nearest_times

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class MixtureDiscreteEulerSolver(Solver):
    def __init__(
        self,
        model: ModelWrapper,
        path: MixtureDiscreteProbPath,
        vocabulary_size: int,
        source_distribution_p: Optional[Tensor] = None,
    ):
        super().__init__()
        self.model = model
        self.path = path
        self.vocabulary_size = vocabulary_size

        if source_distribution_p is not None:
            assert source_distribution_p.shape == torch.Size([vocabulary_size]), (
                f"Source distribution p dimension must match vocabulary size {vocabulary_size}. "
                f"Got {source_distribution_p.shape}."
            )
        self.source_distribution_p = source_distribution_p

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        step_size: Optional[float],
        div_free: Union[float, Callable[[float], float]] = 0.0,
        dtype_categorical: torch.dtype = torch.float32,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        verbose: bool = False,
        **model_extras,
    ) -> Tensor:
        if div_free != 0.0:
            assert self.source_distribution_p is not None, (
                "Source distribution p must be specified when divergence-free term is non-zero."
            )

        time_grid = time_grid.to(device=x_init.device)
        if step_size is None:
            t_discretization = time_grid
            n_steps = len(time_grid) - 1
            t_init = time_grid[0].item()
            t_final = time_grid[-1].item()
        else:
            t_init = time_grid[0].item()
            t_final = time_grid[-1].item()
            assert (t_final - t_init) > step_size, (
                f"Time interval [{t_init}, {t_final}] must be larger than step_size={step_size}."
            )
            n_steps = ceil((t_final - t_init) / step_size)
            t_discretization = torch.tensor(
                [t_init + step_size * i for i in range(n_steps)] + [t_final],
                device=x_init.device,
            )
            if return_intermediates:
                order = torch.argsort(time_grid)
                time_grid = get_nearest_times(time_grid=time_grid, t_discretization=t_discretization)
            else:
                order = None

        x_t = x_init.clone()
        steps_counter = 0
        res = [x_init.clone()] if return_intermediates else []

        if verbose:
            if not TQDM_AVAILABLE:
                raise ImportError("tqdm is required for verbose mode. Please install it.")
            ctx = tqdm(total=t_final, desc=f"NFE: {steps_counter}")
        else:
            ctx = nullcontext()

        with ctx:
            for i in range(n_steps):
                t = t_discretization[i : i + 1]
                h = t_discretization[i + 1 : i + 2] - t_discretization[i : i + 1]

                p_1t = self.model(x=x_t, t=t.repeat(x_t.shape[0]), **model_extras)
                x_1 = categorical(p_1t.to(dtype=dtype_categorical))

                if i == n_steps - 1:
                    x_t = x_1
                else:
                    scheduler_output = self.path.scheduler(t=t)
                    k_t = scheduler_output.alpha_t
                    d_k_t = scheduler_output.d_alpha_t

                    delta_1 = F.one_hot(x_1, num_classes=self.vocabulary_size).to(k_t.dtype)
                    u = d_k_t / (1 - k_t).clamp_min(1e-12) * delta_1

                    div_free_t = div_free(t) if callable(div_free) else div_free
                    if div_free_t > 0:
                        p_0 = self.source_distribution_p[(None,) * x_t.dim()]
                        u = u + div_free_t * d_k_t / (k_t * (1 - k_t)).clamp_min(1e-12) * (
                            (1 - k_t) * p_0 + k_t * delta_1
                        )

                    delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size)
                    u = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(u), u)

                    intensity = u.sum(dim=-1)
                    mask_jump = torch.rand(size=x_t.shape, device=x_t.device) < 1 - torch.exp(-h * intensity)
                    if mask_jump.sum() > 0:
                        x_t[mask_jump] = categorical(u[mask_jump].to(dtype=dtype_categorical))

                steps_counter += 1
                t = t + h

                if return_intermediates and (t in time_grid):
                    res.append(x_t.clone())

                if verbose:
                    ctx.n = t.item()
                    ctx.refresh()
                    ctx.set_description(f"NFE: {steps_counter}")

        if return_intermediates:
            stacked = torch.stack(res, dim=0)
            if step_size is None:
                return stacked
            assert order is not None
            return stacked[order]
        return x_t
