"""One-event-per-item orchestration. No solve driver or recovery policy."""
from dataclasses import replace
import torch
from ..oracle import MVSKOracle, MVSKCache
from ..constraints.feasibility import _inputs
from ..constraints.active_set import update_active_set
from ..diagnostics import simplex_kkt_residual
from ..geometry import build_geometry
from ..geometry import kernels as geometry_kernels
from ..geometry.affine_normal import affine_normal_direction
from ..line_search import exact_mvsk_line_search, LineSearchStatus
from .options import YANDOptions
from .state import YANDState, IterationResult, SolverStatus as S, SolverReason as R, IterationEvent as E
from . import kernels


def _options(options,dtype):
    options=YANDOptions() if options is None else options
    if not isinstance(options,YANDOptions): raise TypeError('options must be YANDOptions')
    options.validate_dtype(dtype)
    return options


def _generator(generator,device,seed):
    if generator is None: return torch.Generator(device=device).manual_seed(seed)
    if not isinstance(generator,torch.Generator) or torch.device(generator.device)!=device:
        raise ValueError('generator must match the input device')
    return generator


def _cache(oracle,x,z):
    return MVSKCache(x,z,oracle._owner,x.shape,x.dtype,x.device)


def _kkt(x,grad,mask,options):
    # Never send a known nonfinite row to a validating public wrapper.
    finite_x=torch.isfinite(x).all(-1); finite_g=torch.isfinite(grad).all(-1)
    xx=kernels.choose(finite_x,x,torch.full_like(x,1/x.shape[-1]))
    gg=kernels.choose(finite_g,grad,torch.zeros_like(grad))
    return simplex_kkt_residual(xx,gg,free_mask=mask,tau=options.tau,
                                rtol=options.kkt_rtol,atol=options.kkt_atol)


def initialize_yand_state(oracle,x0,free_mask=None,*,options=None,generator=None):
    """Create current cache/value/gradient/KKT, with iterations=0.

    Fatal dtype/device/shape/nonfinite input errors raise. Finite but infeasible
    items and finite-input numerical overflow terminate only the affected item.
    Supplied masks may keep released coordinates free at the lower bound.
    """
    if not isinstance(oracle,MVSKOracle): raise TypeError('oracle must be MVSKOracle')
    oracle._validate_tensor(x0,'x0')
    options=_options(options,x0.dtype)
    x,_,mask,_,_=_inputs(x0,x0,free_mask,options.tau)
    # Preserve identity when no leading expansion was necessary.
    if x.shape==x0.shape: x=x0
    gen=_generator(generator,x.device,options.seed)
    z,value,grad,finite=kernels.evaluate(x,oracle.A,oracle.mu,oracle.coefficients)
    kkt=_kkt(x,grad,mask,options)
    valid=finite & kkt.valid
    status=torch.where(valid,torch.where(kkt.kkt_stationary,int(S.CONVERGED_KKT),int(S.RUNNING)),int(S.INVALID_STATE))
    reason=torch.where(~finite,int(R.NAN_DETECTED),torch.where(~kkt.valid,int(R.INVALID_FEASIBILITY),int(R.NONE)))
    count=torch.zeros(x.shape[:-1],device=x.device,dtype=torch.int64)
    scalar=torch.zeros(x.shape[:-1],device=x.device,dtype=x.dtype)
    return YANDState(x=x,free_mask=mask,value=value,grad=grad,cache=_cache(oracle,x,z),
        status=status,reason=reason,active=status==S.RUNNING,iterations=count,
        accepted_steps=count.clone(),face_updates=count.clone(),fallback_count=count.clone(),
        no_progress_count=count.clone(),lambda_reg=torch.full_like(scalar,options.lambda0),
        kkt_residual=torch.where(finite,kkt.kkt_residual,float('nan')),
        last_alpha=scalar,last_step_norm=scalar.clone(),last_objective_change=scalar.clone(),
        face_revision=count.clone(),x_revision=count.clone(),direction_attempts=count.clone(),probe_draws=count.clone(),
        value_refreshes=count+1,gradient_refreshes=count+1,cache_rebuilds=count+1,
        options=options,generator=gen,owner=oracle._owner)


def yand_iteration(oracle,state,*,options=None,generator=None):
    """Execute one release-only or direction/search event per RUNNING item.

    Terminal item tensor fields freeze. If x changes anywhere, a new full
    cache binds the committed x object; terminal cache rows are retained.
    RNG and options are shared services, not per-item tensor fields.
    Low-level total telemetry is unavailable and intentionally None.
    """
    if not isinstance(oracle,MVSKOracle) or not isinstance(state,YANDState): raise TypeError('invalid oracle/state')
    if state.owner is not oracle._owner: raise ValueError('state belongs to another oracle')
    if state.cache.x is not state.x or state.cache.owner is not oracle._owner:
        raise ValueError('state cache must bind current x and oracle')
    if state.cache.shape!=state.x.shape or state.cache.z.shape!=state.x.shape[:-1]+(oracle.T,):
        raise ValueError('state cache metadata mismatch')
    options=_options(state.options if options is None else options,state.x.dtype)
    gen=_generator(state.generator if generator is None else generator,state.x.device,options.seed)
    x,grad,mask=state.x,state.grad,state.free_mask
    active=state.active & (state.status==S.RUNNING)
    flags=torch.zeros_like(active); zero=torch.zeros_like(state.value)
    event=torch.zeros_like(state.iterations); progress=flags.clone()
    if not active.any().item(): return IterationResult(state,event,progress)
    values={name:getattr(state,name) for name in state.__dataclass_fields__}
    status,reason=state.status.clone(),state.reason.clone()
    finite=(torch.isfinite(x).all(-1) & torch.isfinite(grad).all(-1) & torch.isfinite(state.value)
            & torch.isfinite(state.cache.z).all(-1) & torch.isfinite(state.lambda_reg) & (state.lambda_reg>=0))
    current=_kkt(x,grad,mask,options)
    bad=active & (~finite | ~current.valid)
    status=torch.where(bad,int(S.INVALID_STATE),status)
    reason=torch.where(bad,torch.where(~finite,int(R.NAN_DETECTED),int(R.INVALID_FEASIBILITY)),reason)
    ready=active & ~bad
    # Before consuming a new event, honor KKT and already-exhausted budgets.
    status,reason=kernels.finish(status,reason,ready,current,state.iterations,state.face_updates,state.no_progress_count,options)
    ready &= status==S.RUNNING
    release_items=ready & current.face_stationary & ~current.kkt_stationary
    direction_items=ready & ~current.face_stationary
    event=torch.where(release_items,int(E.RELEASE),torch.where(direction_items,int(E.DIRECTION),event))
    attempted=release_items | direction_items
    values['iterations']=state.iterations+attempted.to(torch.int64)
    values['last_alpha']=kernels.choose(attempted,zero,state.last_alpha)
    values['last_step_norm']=kernels.choose(attempted,zero,state.last_step_norm)
    values['last_objective_change']=kernels.choose(attempted,zero,state.last_objective_change)
    safe_x=kernels.choose(torch.isfinite(x).all(-1),x,torch.full_like(x,1/oracle.n))
    safe_grad=kernels.choose(torch.isfinite(grad).all(-1),grad,torch.zeros_like(grad))
    release_result=direction_result=search_result=activation_result=None
    mask_new=mask
    if release_items.any().item():
        release_result=update_active_set(safe_x,safe_grad,mask,phase='release',tau=options.tau,
            active_tol=options.active_tol,kkt_rtol=options.kkt_rtol,kkt_atol=options.kkt_atol)
        invalid=release_items & ~release_result.valid
        status=torch.where(invalid,int(S.INVALID_STATE),status)
        reason=torch.where(invalid,int(R.ACTIVE_SET_FAILURE),reason)
        changed=release_items & release_result.valid & release_result.changed
        mask_new=kernels.choose(changed,release_result.free_mask,mask_new)
        values['face_updates']=state.face_updates+changed.to(torch.int64)
        values['face_revision']=state.face_revision+changed.to(torch.int64)
        no_change=release_items & release_result.valid & ~release_result.changed
        values['no_progress_count']=state.no_progress_count+no_change.to(torch.int64)
    else:
        values['face_updates']=state.face_updates
        values['face_revision']=state.face_revision
    positive=flags.clone()
    new_x=x; new_grad=grad; new_z=state.cache.z; new_value=state.value
    if direction_items.any().item():
        # Audit geometry before build_geometry's batch-wide finite check.
        components=geometry_kernels.geometry(kernels.choose(direction_items,safe_grad,torch.zeros_like(grad)),
                                             mask & direction_items.unsqueeze(-1))
        geom_finite=torch.isfinite(components[0]).all(-1) & torch.isfinite(components[1]) & torch.isfinite(components[2]).all(-1) & torch.isfinite(components[-1])
        invalid=direction_items & ~geom_finite
        status=torch.where(invalid,int(S.DIRECTION_FAILED),status)
        reason=torch.where(invalid,int(R.DIRECTION_NUMERICAL_FAILURE),reason)
        eligible=direction_items & geom_finite
        values['direction_attempts']=state.direction_attempts+direction_items.to(torch.int64)
        if eligible.any().item():
            # Dummy rows use an actual finite eligible point, not necessarily
            # x=0 (whose gradient can overflow even when another point is safe).
            work_x=kernels.anchor_rows(x,eligible)
            work_z,_,work_grad,work_finite=kernels.evaluate(work_x,oracle.A,oracle.mu,oracle.coefficients)
            invalid=eligible & ~work_finite
            status=torch.where(invalid,int(S.DIRECTION_FAILED),status)
            reason=torch.where(invalid,int(R.DIRECTION_NUMERICAL_FAILURE),reason)
            eligible &= work_finite
            work_components=geometry_kernels.geometry(kernels.choose(eligible,work_grad,torch.zeros_like(work_grad)),mask & eligible.unsqueeze(-1))
            work_geom_finite=(torch.isfinite(work_components[0]).all(-1) & torch.isfinite(work_components[1])
                              & torch.isfinite(work_components[2]).all(-1) & torch.isfinite(work_components[-1]))
            invalid=eligible & ~work_geom_finite
            status=torch.where(invalid,int(S.DIRECTION_FAILED),status)
            reason=torch.where(invalid,int(R.DIRECTION_NUMERICAL_FAILURE),reason)
            eligible &= work_geom_finite
            if eligible.any().item():
                # Re-anchor any failed rows without repeating unsafe arithmetic.
                work_x=kernels.anchor_rows(work_x,eligible)
                work_z=kernels.anchor_rows(work_z,eligible)
                work_grad=kernels.anchor_rows(work_grad,eligible)
                geom=build_geometry(kernels.choose(eligible,work_grad,torch.zeros_like(work_grad)),mask & eligible.unsqueeze(-1))
                probes=None
                if options.logdet_mode=='stochastic':
                    raw=torch.randint(0,2,x.shape[:-1]+(options.q,oracle.n),device=x.device,generator=gen).to(x.dtype)*2-1
                    probes=kernels.choose(eligible,raw,torch.zeros_like(raw))
                    values['probe_draws']=state.probe_draws+eligible.to(torch.int64)
                direction_result=affine_normal_direction(oracle,work_x,geom=geom,cache=_cache(oracle,work_x,work_z),
                    lambda0=torch.where(eligible,state.lambda_reg,options.lambda0),lambda_growth=options.lambda_growth,
                    max_regularization_retries=options.max_regularization_retries,logdet_mode=options.logdet_mode,
                    q=options.q,probes=probes,generator=gen,pcg_options=options.pcg_options,descent_factor=options.descent_factor)
                damping_ok=eligible & torch.isfinite(direction_result.lambda_reg) & (direction_result.lambda_reg>=0)
                values['lambda_reg']=kernels.choose(damping_ok,direction_result.lambda_reg,state.lambda_reg)
                valid_direction=eligible & direction_result.success & torch.isfinite(direction_result.direction).all(-1)
                invalid=eligible & ~valid_direction
                status=torch.where(invalid,int(S.DIRECTION_FAILED),status)
                reason=torch.where(invalid,int(R.DIRECTION_NUMERICAL_FAILURE),reason)
                values['fallback_count']=state.fallback_count+(eligible & direction_result.success & direction_result.fallback_used).to(torch.int64)
                if valid_direction.any().item():
                    search_d=kernels.choose(valid_direction,direction_result.direction,torch.zeros_like(x))
                    search_result=exact_mvsk_line_search(oracle,work_x,search_d,free_mask=mask,
                        tau=options.tau,cache=_cache(oracle,work_x,work_z),direction_tol=options.direction_tol,
                        hit_tol=options.hit_tol,interval_tol=options.interval_tol,value_tol=options.value_tol,acceptance_tol=options.acceptance_tol)
                    valid_search=valid_direction & search_result.success
                    invalid=valid_direction & ~search_result.success
                    status=torch.where(invalid,int(S.LINE_SEARCH_FAILED),status)
                    reason=torch.where(invalid,int(R.LINE_SEARCH_NUMERICAL_FAILURE),reason)
                    zero_status=(search_result.status==LineSearchStatus.ZERO_DIRECTION) | (search_result.status==LineSearchStatus.ZERO_CAP) | (search_result.status==LineSearchStatus.ZERO_STEP)
                    zero_step=valid_search & (zero_status | (search_result.alpha<=0))
                    values['no_progress_count']=values['no_progress_count']+zero_step.to(torch.int64)
                    positive=valid_search & ~zero_step & (search_result.alpha>0)
                    if positive.any().item():
                        new_x=kernels.choose(positive,search_result.x_trial,x)
                        # No old x-object cache is reused for this tensor.
                        evaluated_z,evaluated_grad,output_finite=kernels.refresh_gradient(new_x,oracle.A,oracle.mu,oracle.coefficients)
                        new_z=kernels.choose(positive,evaluated_z,state.cache.z)
                        new_grad=kernels.choose(positive,evaluated_grad,grad)
                        new_value=kernels.choose(positive,search_result.trial_value,state.value)
                        values['accepted_steps']=state.accepted_steps+positive.to(torch.int64)
                        values['x_revision']=state.x_revision+positive.to(torch.int64)
                        values['value_refreshes']=state.value_refreshes+positive.to(torch.int64)
                        values['gradient_refreshes']=state.gradient_refreshes+positive.to(torch.int64)
                        # A new batched object rebinds the whole cache; only
                        # changed rows consume logical projection refreshes.
                        values['cache_rebuilds']=state.cache_rebuilds+positive.to(torch.int64)
                        values['last_alpha']=kernels.choose(positive,search_result.alpha,values['last_alpha'])
                        step_norm=torch.linalg.vector_norm(new_x-x,dim=-1)
                        values['last_step_norm']=kernels.choose(positive,step_norm,values['last_step_norm'])
                        values['last_objective_change']=kernels.choose(positive,new_value-state.value,values['last_objective_change'])
                        invalid=positive & ~output_finite
                        status=torch.where(invalid,int(S.INVALID_STATE),status)
                        reason=torch.where(invalid,int(R.NAN_DETECTED),reason)
                        boundary=positive & output_finite & search_result.hits_boundary
                        if boundary.any().item():
                            activation_result=update_active_set(new_x,kernels.choose(output_finite,new_grad,torch.zeros_like(new_grad)),mask,
                                phase='activation',activation_mask=search_result.hit_mask & boundary.unsqueeze(-1),
                                tau=options.tau,active_tol=options.active_tol,kkt_rtol=options.kkt_rtol,kkt_atol=options.kkt_atol)
                            invalid=boundary & ~activation_result.valid
                            status=torch.where(invalid,int(S.INVALID_STATE),status)
                            reason=torch.where(invalid,int(R.ACTIVE_SET_FAILURE),reason)
                            changed=boundary & activation_result.valid & activation_result.changed
                            mask_new=kernels.choose(changed,activation_result.free_mask,mask_new)
                            values['face_updates']=values['face_updates']+changed.to(torch.int64)
                            values['face_revision']=values['face_revision']+changed.to(torch.int64)
    final=_kkt(new_x,new_grad,mask_new,options)
    same_face=~(mask_new!=mask).any(-1)
    eps=torch.finfo(x.dtype).eps
    x_tol=64*eps*x.abs().amax(-1).clamp_min(1)
    f_tol=128*eps*torch.maximum(state.value.abs(),new_value.abs()).clamp_min(1)
    progress=positive & (values['last_step_norm']>x_tol) & ((state.value-new_value>f_tol) | (same_face & (current.kkt_residual-final.kkt_residual>current.stopping_threshold)))
    values['no_progress_count']=kernels.choose(positive,torch.where(progress,0,values['no_progress_count']+1),values['no_progress_count'])
    can_finish=attempted & (status==S.RUNNING)
    status,reason=kernels.finish(status,reason,can_finish,final,values['iterations'],values['face_updates'],values['no_progress_count'],options)
    values.update(x=new_x,free_mask=mask_new,value=new_value,grad=new_grad,
        cache=state.cache if new_x is x else _cache(oracle,new_x,new_z),status=status,reason=reason,
        active=status==S.RUNNING,kkt_residual=kernels.choose(active,final.kkt_residual,state.kkt_residual),options=options,generator=gen)
    return IterationResult(YANDState(**values),event,progress,current,final,release_result,direction_result,search_result,activation_result)
