"""Final per-item results and detached lightweight eager diagnostics."""
from dataclasses import dataclass, fields
import torch
from .state import YANDState, SolverStatus


@dataclass(frozen=True)
class YANDHistoryEntry:
    """One post-driver-iteration snapshot; no cache or autograd graph."""
    x: torch.Tensor
    value: torch.Tensor
    status: torch.Tensor
    reason: torch.Tensor
    kkt_residual: torch.Tensor
    iterations: torch.Tensor
    accepted_steps: torch.Tensor
    face_updates: torch.Tensor
    last_alpha: torch.Tensor

    @classmethod
    def from_state(cls,state):
        return cls(**{f.name:getattr(state,f.name).detach().clone() for f in fields(cls)})


@dataclass(frozen=True)
class YANDResult:
    """Result tensor fields reference final_state; history owns detached copies.

    All scalar flags/counts preserve the input batch shape. Numerical terminal
    failures are per item; programming errors still raise. Unknown low-level
    totals remain None rather than extrapolating the last nested solve.
    """
    x: torch.Tensor
    value: torch.Tensor
    grad: torch.Tensor
    free_mask: torch.Tensor
    status: torch.Tensor
    reason: torch.Tensor
    converged: torch.Tensor
    iterations: torch.Tensor
    accepted_steps: torch.Tensor
    face_updates: torch.Tensor
    fallback_count: torch.Tensor
    no_progress_count: torch.Tensor
    kkt_residual: torch.Tensor
    lambda_reg: torch.Tensor
    last_alpha: torch.Tensor
    last_step_norm: torch.Tensor
    last_objective_change: torch.Tensor
    final_state: YANDState
    history: tuple[YANDHistoryEntry,...] | None
    total_hvp: None = None
    total_third: None = None
    total_pcg_iterations: None = None
    total_value_evaluations: None = None
    total_gradient_evaluations: None = None

    @property
    def all_converged(self):
        """Convenience bool; synchronizes when tensors are on CUDA."""
        return bool(self.converged.all().item())

    @classmethod
    def from_state(cls,state,history=None):
        values={f.name:getattr(state,f.name) for f in fields(cls)
                if f.name not in ('converged','final_state','history')}
        return cls(**values,converged=state.status==SolverStatus.CONVERGED_KKT,
                   final_state=state,history=history)
