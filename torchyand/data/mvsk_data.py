"""One-time centering of sample returns, preserving device and autograd."""
from dataclasses import dataclass
import torch


SUPPORTED_DTYPES = (torch.float32, torch.float64)


@dataclass(frozen=True)
class MVSKData:
    """Shared data; A/mu must not be modified during oracle/cache use."""
    A: torch.Tensor
    mu: torch.Tensor
    T: int
    n: int


def prepare_mvsk_data(R: torch.Tensor, center_dtype=torch.float64,
                      output_dtype=None) -> MVSKData:
    """Center finite [T,n] returns once. T,n must be positive.

    Only float32/64 are accepted. None center_dtype uses R.dtype; None
    output_dtype preserves R.dtype. Casting stays on R.device and does not
    detach. No column standardization. Finite-input arithmetic overflow raises
    FloatingPointError, rather than returning nonfinite data.
    """
    if not isinstance(R, torch.Tensor):
        raise TypeError("R must be a torch.Tensor")
    if R.dtype not in SUPPORTED_DTYPES:
        raise TypeError("Only float32 and float64 are supported")
    if R.ndim != 2 or min(R.shape) < 1:
        raise ValueError("R must have nonempty shape [T,n]")
    center_dtype = R.dtype if center_dtype is None else center_dtype
    output_dtype = R.dtype if output_dtype is None else output_dtype
    if center_dtype not in SUPPORTED_DTYPES or output_dtype not in SUPPORTED_DTYPES:
        raise TypeError("center_dtype/output_dtype must be float32 or float64")
    if not torch.isfinite(R).all().item():
        raise ValueError("R must be finite")
    centered_input = R.to(dtype=center_dtype)
    mu = centered_input.mean(dim=0)
    A = (centered_input-mu).to(dtype=output_dtype)
    mu = mu.to(dtype=output_dtype)
    if not (torch.isfinite(A).all() & torch.isfinite(mu).all()).item():
        raise FloatingPointError("MVSK centering or output cast overflowed")
    return MVSKData(A, mu, R.shape[0], R.shape[1])
