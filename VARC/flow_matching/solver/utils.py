from __future__ import annotations

import torch
from torch import Tensor


def get_nearest_times(time_grid: Tensor, t_discretization: Tensor) -> Tensor:
    idx = torch.cdist(time_grid[:, None], t_discretization[:, None]).argmin(dim=1)
    return t_discretization[idx]
