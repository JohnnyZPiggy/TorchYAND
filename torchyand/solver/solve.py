"""Eager driver for the frozen single-event state machine, no new numerics."""
from dataclasses import replace
import torch
from ..oracle import MVSKOracle
from .options import YANDOptions
from .iteration import initialize_yand_state, yand_iteration
from .state import SolverStatus, SolverReason
from .result import YANDResult, YANDHistoryEntry


def _configuration(oracle,options,seed,generator):
    if not isinstance(oracle,MVSKOracle): raise TypeError('oracle must be MVSKOracle')
    options=YANDOptions() if options is None else options
    if not isinstance(options,YANDOptions): raise TypeError('options must be YANDOptions')
    if options.max_iter<1: raise ValueError('driver max_iter must be >=1')
    if options.stall_patience<1 or options.max_face_updates<0: raise ValueError('invalid driver budgets')
    options.validate_dtype(oracle.A.dtype)
    if seed is not None and generator is not None: raise ValueError('seed and generator are mutually exclusive')
    if seed is not None and (isinstance(seed,bool) or not isinstance(seed,int)):
        raise TypeError('seed must be an integer')
    if generator is not None:
        if not isinstance(generator,torch.Generator): raise TypeError('generator must be torch.Generator')
        if torch.device(generator.device)!=oracle.A.device: raise ValueError('generator must match oracle device')
    return options


def solve_yand(oracle,x0,*,free_mask=None,options=None,generator=None,seed=None,return_history=False):
    """Run initialization and frozen yand_iteration until every item stops.

    Does not duplicate KKT/direction/search/face/stall logic. The internal
    max_iter+1 guard diagnoses an orchestration invariant failure, not MAX_ITER.
    Explicit generators advance in place. Otherwise each call owns a fresh
    device-local generator (explicit seed or options.seed), never global RNG.
    History contains one detached snapshot per actual driver iteration; an
    initially terminal batch has empty history when requested. CUDA outer
    active checks intentionally synchronize. Options remain fixed throughout.
    """
    options=_configuration(oracle,options,seed,generator)
    if not isinstance(return_history,bool): raise TypeError('return_history must be bool')
    if seed is not None:
        # Oracle and x0 must share device, checked by frozen initialization.
        generator=torch.Generator(device=oracle.A.device).manual_seed(seed)
    state=initialize_yand_state(oracle,x0,free_mask,options=options,generator=generator)
    snapshots=[] if return_history else None
    loops=0
    max_driver_loops=options.max_iter+1
    while state.active.any().item():
        iteration_result=yand_iteration(oracle,state,options=options,generator=state.generator)
        state=iteration_result.state
        loops+=1
        if loops>=max_driver_loops and state.active.any().item():
            # Only still-active rows change; all terminal rows stay frozen.
            state=replace(state,
                status=torch.where(state.active,int(SolverStatus.INVALID_STATE),state.status),
                reason=torch.where(state.active,int(SolverReason.DRIVER_GUARD_EXCEEDED),state.reason),
                active=torch.zeros_like(state.active))
        if snapshots is not None: snapshots.append(YANDHistoryEntry.from_state(state))
    return YANDResult.from_state(state,None if snapshots is None else tuple(snapshots))


class YANDMVSK:
    """Configuration-only facade; solve delegates directly to solve_yand.

    Seed/default RNG calls restart the configured stream per solve. An
    explicitly supplied generator is caller-owned and advances across solves.
    No hidden iteration state, alternate mathematics or recovery policy.
    """
    def __init__(self,oracle,*,options=None,seed=None,generator=None):
        self.options=_configuration(oracle,options,seed,generator)
        self.oracle,self.seed,self.generator=oracle,seed,generator

    def solve(self,x0,free_mask=None,return_history=False):
        return solve_yand(self.oracle,x0,free_mask=free_mask,options=self.options,
                          seed=self.seed,generator=self.generator,return_history=return_history)
