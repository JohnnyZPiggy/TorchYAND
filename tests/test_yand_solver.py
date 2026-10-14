"""Public driver contracts and genuine very-small-n integration problems."""
from dataclasses import replace
import importlib
import pytest
import torch
from torchyand.data import MVSKData
from torchyand.oracle import MVSKOracle
from torchyand.diagnostics import simplex_kkt_residual
from torchyand.solver import (solve_yand,YANDMVSK,YANDOptions,YANDResult,
    SolverStatus as S,SolverReason as R)
from torchyand.solver import iteration as iteration_module

driver=importlib.import_module('torchyand.solver.solve')


@pytest.fixture(params=[torch.float32,torch.float64])
def dtype(request): return request.param


def oracle(dtype,linear=False,n=2):
    if linear:
        A=torch.zeros(2,n,dtype=dtype); mu=torch.arange(n,dtype=dtype); c=[1,0,0,0]
    else:
        A=torch.zeros(2,n,dtype=dtype); A[:,0]=torch.tensor([-1,1],dtype=dtype)
        mu=torch.zeros(n,dtype=dtype); mu[0]=.5; c=[1,1,0,0]
    return MVSKOracle(MVSKData(A,mu,2,n),c)


def point(dtype,t): return torch.tensor([t,1-t],dtype=dtype)


def options(**kw): return YANDOptions(logdet_mode='exact',**kw)


def verify_end_to_end(o,x0,result,opts):
    eps=torch.finfo(x0.dtype).eps
    assert not result.final_state.active.any()
    assert (result.x>=opts.tau-64*eps).all()
    assert ((result.x.sum(-1)-1).abs()<=64*eps*result.x.abs().sum(-1).clamp_min(1)).all()
    initial=o.value(x0)
    allowance=256*eps*torch.maximum(initial.abs(),result.value.abs()).clamp_min(1)
    assert (result.value<=initial+allowance).all()
    kkt=simplex_kkt_residual(result.x,o.gradient(result.x),free_mask=result.free_mask,
                           tau=opts.tau,rtol=opts.kkt_rtol,atol=opts.kkt_atol)
    assert (kkt.valid | ~result.converged).all()
    assert (kkt.kkt_stationary | ~result.converged).all()
    assert not (result.reason==R.DRIVER_GUARD_EXCEEDED).any()


def test_initial_optimum_immediate(dtype,monkeypatch):
    o=oracle(dtype); x=point(dtype,.25)
    def forbidden(*args,**kw): raise AssertionError('no driver iteration at initial optimum')
    monkeypatch.setattr(driver,'yand_iteration',forbidden)
    result=solve_yand(o,x,options=options(),return_history=True)
    assert result.status==S.CONVERGED_KKT and result.converged and result.iterations==0
    assert result.all_converged and result.history==()


@pytest.mark.parametrize('mode',['exact','stochastic'])
def test_real_quadratic_convergence(dtype,mode):
    o=oracle(dtype); x=point(dtype,.1); opts=YANDOptions(logdet_mode=mode,max_iter=10)
    r=solve_yand(o,x,options=opts,seed=167)
    assert r.status==S.CONVERGED_KKT and r.converged and r.accepted_steps==1
    torch.testing.assert_close(r.x,point(dtype,.25))
    assert r.history is None
    verify_end_to_end(o,x,r,opts)


def test_release_then_search_and_boundary(dtype):
    o=oracle(dtype,linear=True); x=point(dtype,1); opts=options()
    r=solve_yand(o,x,options=opts,return_history=True)
    assert r.converged and r.iterations==2 and r.accepted_steps==1 and r.face_updates==2
    assert len(r.history)==2
    torch.testing.assert_close(r.history[0].x,x,rtol=0,atol=0)
    assert r.history[0].accepted_steps==0 and r.history[0].face_updates==1
    assert r.free_mask.tolist()==[False,True] and r.x[0].abs()<=64*torch.finfo(dtype).eps
    verify_end_to_end(o,x,r,opts)


@pytest.mark.parametrize('quartic',[False,True])
@pytest.mark.parametrize('mode',['exact','stochastic'])
def test_real_symmetric_three_asset_reference(dtype,quartic,mode):
    n=3; eye=torch.eye(n,dtype=dtype); A=torch.cat((eye,-eye),0)
    o=MVSKOracle(MVSKData(A,torch.zeros(n,dtype=dtype),2*n,n),[0,1,.1 if quartic else 0,.05 if quartic else 0])
    x=torch.tensor([.1,.2,.7],dtype=dtype)
    opts=YANDOptions(logdet_mode=mode,max_iter=30)
    r=solve_yand(o,x,options=opts,seed=123)
    assert r.status in (S.CONVERGED_KKT,S.STALLED,S.MAX_ITER)
    verify_end_to_end(o,x,r,opts)
    optimum=torch.full_like(x,1/n)
    assert r.value>=o.value(optimum)-512*torch.finfo(dtype).eps
    if not quartic:
        assert r.converged
        torch.testing.assert_close(r.x,optimum)
    if r.converged:
        torch.testing.assert_close(r.value,o.value(optimum))


def test_real_four_asset_linear_faces(dtype):
    o=oracle(dtype,linear=True,n=4); x=torch.full((4,),.25,dtype=dtype); opts=options(max_iter=20)
    r=solve_yand(o,x,options=opts,return_history=True)
    assert r.converged and r.face_updates>=1
    torch.testing.assert_close(r.x,torch.tensor([0,0,0,1],dtype=dtype))
    verify_end_to_end(o,x,r,opts)


def install_zero_search(monkeypatch):
    original=iteration_module.exact_mvsk_line_search
    def zero(*args,**kw):
        r=original(*args,**kw)
        from torchyand.line_search import LineSearchStatus
        return replace(r,alpha=torch.zeros_like(r.alpha),x_trial=args[1],trial_value=r.current_value,
                       status=torch.full_like(r.status,int(LineSearchStatus.ZERO_STEP)),hits_boundary=torch.zeros_like(r.hits_boundary))
    monkeypatch.setattr(iteration_module,'exact_mvsk_line_search',zero)


@pytest.mark.parametrize('kind',['max_iter','stall'])
def test_budgets_and_stall(dtype,kind,monkeypatch):
    install_zero_search(monkeypatch)
    opts=options(max_iter=1,stall_patience=3) if kind=='max_iter' else options(max_iter=10,stall_patience=2)
    r=solve_yand(oracle(dtype),point(dtype,.1),options=opts,return_history=True)
    assert r.status==(S.MAX_ITER if kind=='max_iter' else S.STALLED)
    assert r.iterations==(1 if kind=='max_iter' else 2)
    assert not r.converged and not r.all_converged
    assert len(r.history)==int(r.iterations)


@pytest.mark.parametrize('kind',['direction','search'])
def test_failure_propagation(dtype,kind,monkeypatch):
    name='affine_normal_direction' if kind=='direction' else 'exact_mvsk_line_search'
    original=getattr(iteration_module,name)
    def failed(*args,**kw):
        r=original(*args,**kw); return replace(r,success=torch.zeros_like(r.success))
    monkeypatch.setattr(iteration_module,name,failed)
    r=solve_yand(oracle(dtype),point(dtype,.1),options=options())
    assert r.status==(S.DIRECTION_FAILED if kind=='direction' else S.LINE_SEARCH_FAILED)
    assert r.iterations==1 and not r.converged


def test_mixed_terminal_freeze_across_driver_loops(dtype,monkeypatch):
    o=oracle(dtype)
    x=torch.tensor([[.25,.75],[0,1],[.1,.9],[.4,.6]],dtype=dtype)
    mask=torch.tensor([[True,True],[False,True],[True,True],[True,True]])
    original=iteration_module.affine_normal_direction
    def partial(*args,**kw):
        r=original(*args,**kw); success=r.success.clone(); success[3]=False
        return replace(r,success=success)
    monkeypatch.setattr(iteration_module,'affine_normal_direction',partial)
    r=solve_yand(o,x,free_mask=mask,options=options(),return_history=True)
    assert r.status.tolist()==[S.CONVERGED_KKT,S.CONVERGED_KKT,S.CONVERGED_KKT,S.DIRECTION_FAILED]
    assert r.iterations.tolist()==[0,2,1,1]
    assert r.converged.tolist()==[True,True,True,False] and not r.all_converged
    assert len(r.history)==2
    for field in ('x','value','status','reason','kkt_residual','iterations','accepted_steps','face_updates','last_alpha'):
        a=getattr(r.history[0],field); b=getattr(r.history[1],field)
        keep=torch.tensor([True,False,True,True])
        torch.testing.assert_close(a[keep],b[keep],rtol=0,atol=0,equal_nan=True)


def test_damping_inherited_by_driver(dtype,monkeypatch):
    install_zero_search(monkeypatch)
    original=iteration_module.affine_normal_direction; seen=[]
    def bumped(*args,**kw):
        seen.append(kw['lambda0'].clone()); r=original(*args,**kw)
        return replace(r,lambda_reg=r.lambda_reg+1)
    monkeypatch.setattr(iteration_module,'affine_normal_direction',bumped)
    r=solve_yand(oracle(dtype),point(dtype,.1),options=options(stall_patience=2))
    assert len(seen)==2
    torch.testing.assert_close(seen[1],seen[0]+1)
    torch.testing.assert_close(r.lambda_reg,seen[0]+2)


def test_rng_reproducibility_no_global_mutation_and_wrapper(dtype):
    o=oracle(dtype,linear=True,n=3); x=torch.tensor([.2,.3,.5],dtype=dtype)
    opts=YANDOptions()
    global_state=torch.random.get_rng_state().clone()
    a=solve_yand(o,x,options=opts,seed=512)
    b=solve_yand(o,x,options=opts,seed=512)
    c=YANDMVSK(o,options=opts,seed=512).solve(x)
    for r in (b,c):
        torch.testing.assert_close(a.x,r.x,rtol=0,atol=0)
        torch.testing.assert_close(a.status,r.status,rtol=0,atol=0)
        assert torch.equal(a.final_state.generator.get_state(),r.final_state.generator.get_state())
    assert torch.equal(global_state,torch.random.get_rng_state())
    # Default solver-owned RNG also leaves global RNG unchanged.
    solve_yand(o,x,options=opts)
    assert torch.equal(global_state,torch.random.get_rng_state())
    gen=torch.Generator().manual_seed(512); before=gen.get_state().clone()
    d=solve_yand(o,x,options=opts,generator=gen)
    assert d.final_state.generator is gen and not torch.equal(before,gen.get_state())


def test_history_detached_snapshots_and_result_fields(dtype):
    o=oracle(dtype,linear=True); x=point(dtype,1).requires_grad_()
    r=solve_yand(o,x,options=options(),return_history=True)
    assert isinstance(r,YANDResult)
    for name in ('x','value','grad','free_mask','status','reason','iterations','accepted_steps','face_updates','fallback_count','no_progress_count','kkt_residual','lambda_reg','last_alpha','last_step_norm','last_objective_change'):
        assert getattr(r,name) is getattr(r.final_state,name)
    assert r.total_hvp is None and r.total_pcg_iterations is None
    snapshots=r.history
    assert len(snapshots)==2 and snapshots[-1].iterations==r.iterations
    assert not snapshots[0].x.requires_grad and snapshots[0].x.grad_fn is None
    assert snapshots[-1].x.data_ptr()!=r.x.data_ptr()
    assert snapshots[0].x.data_ptr()!=snapshots[-1].x.data_ptr()
    saved=r.x.detach().clone(); later=snapshots[-1].x.clone()
    snapshots[0].x.add_(9)
    torch.testing.assert_close(r.x,saved,rtol=0,atol=0)
    torch.testing.assert_close(snapshots[-1].x,later,rtol=0,atol=0)
    assert not hasattr(snapshots[0],'cache') and not hasattr(snapshots[0],'probes')


def test_guard_diagnoses_invariant_and_preserves_terminal(dtype,monkeypatch):
    o=oracle(dtype); x=torch.tensor([[.25,.75],[.1,.9]],dtype=dtype); calls=[0]
    def broken(oracle,state,**kw):
        from torchyand.solver import IterationResult
        calls[0]+=1
        return IterationResult(state,torch.zeros_like(state.iterations),torch.zeros_like(state.active))
    monkeypatch.setattr(driver,'yand_iteration',broken)
    r=solve_yand(o,x,options=options(max_iter=2),return_history=True)
    assert calls[0]==3 and len(r.history)==3
    assert r.status.tolist()==[S.CONVERGED_KKT,S.INVALID_STATE]
    assert r.reason.tolist()==[R.NONE,R.DRIVER_GUARD_EXCEEDED]
    assert r.iterations.tolist()==[0,0]
    assert torch.equal(r.history[-1].status,r.status)
    assert not r.final_state.active.any()


def test_partial_initial_invalid_state(dtype):
    o=oracle(dtype); x=torch.tensor([[.25,.75],[.4,.4],[.1,.9]],dtype=dtype)
    r=solve_yand(o,x,options=options())
    assert r.status.tolist()==[S.CONVERGED_KKT,S.INVALID_STATE,S.CONVERGED_KKT]
    assert r.iterations.tolist()==[0,0,1]
    torch.testing.assert_close(r.x[1],x[1],rtol=0,atol=0)


@pytest.mark.parametrize('kind',['both_rng','max_zero','seed_type','dtype','shape','history_type'])
def test_api_errors(kind):
    o=oracle(torch.float64); x=point(torch.float64,.1); kwargs={}
    if kind=='both_rng': kwargs.update(seed=1,generator=torch.Generator())
    if kind=='max_zero': kwargs['options']=options(max_iter=0)
    if kind=='seed_type': kwargs['seed']=1.5
    if kind=='dtype': x=x.float()
    if kind=='shape': x=x[:1]
    if kind=='history_type': kwargs['return_history']=1
    with pytest.raises((ValueError,TypeError)): solve_yand(o,x,**kwargs)
    if kind=='both_rng':
        with pytest.raises(ValueError): YANDMVSK(o,seed=1,generator=torch.Generator())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_parity(dtype):
    o=oracle(dtype); x=point(dtype,.1); opts=options()
    gpu=MVSKOracle(MVSKData(o.A.cuda(),o.mu.cuda(),o.T,o.n),o.coefficients.cuda())
    a=solve_yand(o,x,options=opts,seed=123)
    b=solve_yand(gpu,x.cuda(),options=opts,seed=123)
    eps=torch.finfo(dtype).eps
    torch.testing.assert_close(a.x,b.x.cpu(),rtol=512*eps,atol=512*eps)
    torch.testing.assert_close(a.value,b.value.cpu(),rtol=512*eps,atol=512*eps)
    assert a.status==b.status.cpu() and a.converged==b.converged.cpu()
