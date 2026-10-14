"""Validated orchestration policy; no mathematical solver implementation."""
from dataclasses import dataclass, field
import math
import numbers
from ..geometry.affine_normal import _options


@dataclass(frozen=True)
class YANDOptions:
    max_iter: int = 200
    stall_patience: int = 5
    max_face_updates: int | None = None
    kkt_rtol: float | None = None
    kkt_atol: float = 0.0
    tau: float = 0.0
    lambda0: float = 1e-6
    lambda_growth: float = 10.0
    max_regularization_retries: int = 5
    logdet_mode: str = 'stochastic'
    q: int = 4
    pcg_options: dict = field(default_factory=dict)
    descent_factor: float = 64.0
    active_tol: float | None = None
    direction_tol: float | None = None
    hit_tol: float | None = None
    interval_tol: float | None = None
    value_tol: float | None = None
    acceptance_tol: float | None = None
    seed: int = 0

    def __post_init__(self):
        if self.max_face_updates is None:
            object.__setattr__(self,'max_face_updates',2*self.max_iter)
        for name in ('max_iter','max_face_updates','max_regularization_retries'):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,int) or v<0: raise ValueError(f'{name} must be a nonnegative integer')
        for name in ('stall_patience','q'):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,int) or v<1: raise ValueError(f'{name} must be a positive integer')
        for name in ('kkt_rtol','kkt_atol','tau','lambda0','active_tol','direction_tol','hit_tol','interval_tol','value_tol','acceptance_tol'):
            v=getattr(self,name)
            if v is not None and (not isinstance(v,numbers.Real) or not math.isfinite(v) or v<0): raise ValueError(f'{name} must be finite and nonnegative')
        if not math.isfinite(self.lambda_growth) or self.lambda_growth<=1: raise ValueError('lambda_growth must exceed one')
        if not math.isfinite(self.descent_factor) or self.descent_factor<=0: raise ValueError('descent_factor must be positive')
        if self.logdet_mode not in ('exact','stochastic'): raise ValueError('invalid logdet_mode')
        if isinstance(self.seed,bool) or not isinstance(self.seed,int): raise ValueError('seed must be an integer')
        object.__setattr__(self,'pcg_options',dict(self.pcg_options))

    def validate_dtype(self,dtype):
        _options(self.pcg_options,dtype)
