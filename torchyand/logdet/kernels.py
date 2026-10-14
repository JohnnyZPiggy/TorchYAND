"""Pure tensor probe/contraction algebra using frozen third-action kernels."""
import torch
from ..geometry.kernels import level_tangent
from ..oracle.kernels import third_action


def project_probes(probes, free_mask, nu, usable):
    xi = level_tangent(probes, free_mask.unsqueeze(-2), nu.unsqueeze(-2))
    return torch.where(usable[..., None, None], xi, 0)


def stochastic_contraction(z, y, xi, A, c, free_mask, nu):
    raw = third_action(z.unsqueeze(-2), y, xi, A, c)
    contribution = level_tangent(raw, free_mask.unsqueeze(-2), nu.unsqueeze(-2))
    correction = level_tangent(contribution.mean(dim=-2), free_mask, nu)
    finite = (torch.isfinite(raw).all(dim=-1).all(dim=-1)
              & torch.isfinite(contribution).all(dim=-1).all(dim=-1)
              & torch.isfinite(correction).all(dim=-1))
    return correction, finite
