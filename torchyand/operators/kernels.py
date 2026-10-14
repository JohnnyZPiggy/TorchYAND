"""Unchecked projected Hessian composition of frozen tensor kernels."""
import torch
from ..geometry.kernels import level_tangent
from ..oracle.kernels import hvp


def projected_hessian(v, z, A, c, free_mask, nu, free_count, level_regular, damping):
    nonempty = (free_count-1-level_regular.to(free_count.dtype)) > 0
    q = torch.where(nonempty.unsqueeze(-1), level_tangent(v, free_mask, nu), 0)
    hq = hvp(z, q, A, c)
    result = level_tangent(hq, free_mask, nu)+damping.unsqueeze(-1)*q
    return torch.where(nonempty.unsqueeze(-1), result, 0)
