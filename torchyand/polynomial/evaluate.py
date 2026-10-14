"""Horner evaluation for ascending quartic coefficients [..., 5].

Inputs to these internal helpers are validated by the interval minimizer.
The last dimension of x is a candidate axis; leading dimensions broadcast.
"""

import torch


def quartic_shifted_horner(coeff: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    _, a1, a2, a3, a4 = (v.unsqueeze(-1) for v in coeff.unbind(-1))
    return x * (a1 + x * (a2 + x * (a3 + x * a4)))


def quartic_horner(coeff: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return coeff[..., 0].unsqueeze(-1) + quartic_shifted_horner(coeff, x)
