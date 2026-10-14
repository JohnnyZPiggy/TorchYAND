"""Single-event orchestration, integration and isolated failure semantics."""
from dataclasses import replace
import pytest
import torch
from torchyand.data import MVSKData, prepare_mvsk_data
from torchyand.oracle import MVSKOracle
from torchyand.solver import (YANDOptions,initialize_yand_state,yand_iteration,
    SolverStatus as S,SolverReason as R,IterationEvent as E)
from torchyand.solver import iteration as module
from torchyand.line_search import LineSearchStatus as LS


@pytest.fixture(params=[torch.float32,torch.float64])
def dtype(request): return request.param


def oracle(dtype,linear=False,n=2):
    if linear:
        A=torch.zeros(2,n,dtype=dtype); mu=torch.arange(n,dtype=dtype)
        c=[1,0,0,0]
    else:
        A=torch.zeros(2,n,dtype=dtype); A[:,0]=torch.tensor([-1,1],dtype=dtype)
        mu=torch.zeros(n,dtype=dtype); mu[0]=.5; c=[1,1,0,0]
    return MVSKOracle(MVSKData(A,mu,2,n),c)


def point(dtype,t): return torch.tensor([t,1-t],dtype=dtype)


def opts(**kwargs): return YANDOptions(logdet_mode='exact',**kwargs)


def test_initial_converged(dtype):
    o=oracle(dtype); x=point(dtype,.25)
    s=initialize_yand_state(o,x,options=opts())
    assert s.status==S.CONVERGED_KKT and s.iterations==0 and not s.active
    assert s.cache.x is x
    result=yand_iteration(o,s)
    assert result.state is s and result.event==E.NONE


def test_release_only_and_bound_cache(dtype,monkeypatch):
    o=oracle(dtype,linear=True); x=point(dtype,1)
    s=initialize_yand_state(o,x,options=opts())
    s=replace(s,no_progress_count=s.no_progress_count+2)
    def forbidden(*a,**kw): raise AssertionError('release must not run direction')
    monkeypatch.setattr(module,'affine_normal_direction',forbidden)
    result=yand_iteration(o,s); new=result.state
    assert result.event==E.RELEASE and new.iterations==1 and new.face_updates==1
    assert new.free_mask.tolist()==[True,True] and new.face_revision==1
    assert new.x is x and new.cache is s.cache and new.grad is s.grad and new.value is s.value
    assert new.no_progress_count==2 and new.probe_draws==0
    assert result.final_kkt.valid and new.status==S.RUNNING


def test_positive_step_fresh_cache_and_progress(dtype):
    o=oracle(dtype); s=initialize_yand_state(o,point(dtype,.1),options=opts())
    s=replace(s,no_progress_count=s.no_progress_count+3)
    r=yand_iteration(o,s); new=r.state
    assert r.event==E.DIRECTION and new.accepted_steps==1 and new.iterations==1
    assert new.x is not s.x and new.cache.x is new.x and new.cache is not s.cache
    assert new.x_revision==1 and new.gradient_refreshes==2 and new.no_progress_count==0
    assert r.measurable_progress and new.last_objective_change<0
    torch.testing.assert_close(new.x,point(dtype,.25))
    torch.testing.assert_close(new.grad,o.gradient(new.x))
    torch.testing.assert_close(new.value,o.value(new.x))
    assert new.status==S.CONVERGED_KKT
    for field in ('total_hvp','total_third','total_pcg_iterations','total_value_evaluations','total_gradient_evaluations'):
        assert getattr(new,field) is None


def test_boundary_activation_and_final_kkt_priority(dtype):
    o=oracle(dtype,linear=True)
    s=initialize_yand_state(o,point(dtype,.5),options=opts(max_iter=1,max_face_updates=1))
    r=yand_iteration(o,s); new=r.state
    assert new.iterations==1 and new.face_updates==1 and new.face_revision==1
    assert r.line_search_result.hits_boundary and r.activation_result.changed
    assert new.free_mask.tolist()==[False,True]
    assert new.status==S.CONVERGED_KKT and new.reason==R.NONE


def test_fallback_allowed(dtype,monkeypatch):
    o=oracle(dtype); s=initialize_yand_state(o,point(dtype,.1),options=opts())
    original=module.affine_normal_direction
    def fallback(*args,**kw):
        r=original(*args,**kw)
        return replace(r,fallback_used=torch.ones_like(r.success))
    monkeypatch.setattr(module,'affine_normal_direction',fallback)
    r=yand_iteration(o,s)
    assert r.line_search_result is not None and r.state.accepted_steps==1 and r.state.fallback_count==1


def test_direction_failure(dtype,monkeypatch):
    o=oracle(dtype); s=initialize_yand_state(o,point(dtype,.1),options=opts())
    original=module.affine_normal_direction
    def failure(*args,**kw):
        r=original(*args,**kw); return replace(r,success=torch.zeros_like(r.success))
    def forbidden(*a,**kw): raise AssertionError('failed direction must not search')
    monkeypatch.setattr(module,'affine_normal_direction',failure)
    monkeypatch.setattr(module,'exact_mvsk_line_search',forbidden)
    r=yand_iteration(o,s)
    assert r.state.status==S.DIRECTION_FAILED and r.state.reason==R.DIRECTION_NUMERICAL_FAILURE
    assert r.state.iterations==1 and r.state.x is s.x and torch.equal(r.state.free_mask,s.free_mask)


def test_search_failure(dtype,monkeypatch):
    o=oracle(dtype); s=initialize_yand_state(o,point(dtype,.1),options=opts())
    original=module.exact_mvsk_line_search
    def failure(*args,**kw):
        r=original(*args,**kw); return replace(r,success=torch.zeros_like(r.success))
    monkeypatch.setattr(module,'exact_mvsk_line_search',failure)
    r=yand_iteration(o,s)
    assert r.state.status==S.LINE_SEARCH_FAILED and r.state.iterations==1
    assert r.state.x is s.x and r.state.cache is s.cache


def install_zero(monkeypatch,code):
    original=module.exact_mvsk_line_search
    def zero(*args,**kw):
        r=original(*args,**kw)
        return replace(r,alpha=torch.zeros_like(r.alpha),x_trial=args[1],trial_value=r.current_value,
                       status=torch.full_like(r.status,int(code)),hits_boundary=torch.zeros_like(r.hits_boundary))
    monkeypatch.setattr(module,'exact_mvsk_line_search',zero)


@pytest.mark.parametrize('code',[LS.ZERO_DIRECTION,LS.ZERO_CAP,LS.ZERO_STEP])
def test_zero_stall(dtype,code,monkeypatch):
    o=oracle(dtype); s=initialize_yand_state(o,point(dtype,.1),options=opts(stall_patience=2))
    install_zero(monkeypatch,code)
    r=yand_iteration(o,s)
    assert r.state.status==S.RUNNING and r.state.no_progress_count==1
    assert r.state.x is s.x and r.state.cache is s.cache
    r2=yand_iteration(o,r.state)
    assert r2.state.status==S.STALLED and r2.state.reason==R.ZERO_PROGRESS and r2.state.iterations==2
    assert r2.state.accepted_steps==0


def test_damping_inheritance(dtype,monkeypatch):
    o=oracle(dtype); s=initialize_yand_state(o,point(dtype,.1),options=opts())
    install_zero(monkeypatch,LS.ZERO_STEP)
    original=module.affine_normal_direction; seen=[]
    def damping(*args,**kw):
        seen.append(kw['lambda0'].clone()); r=original(*args,**kw)
        return replace(r,lambda_reg=r.lambda_reg+1)
    monkeypatch.setattr(module,'affine_normal_direction',damping)
    a=yand_iteration(o,s).state; b=yand_iteration(o,a).state
    torch.testing.assert_close(seen[0],s.lambda_reg)
    torch.testing.assert_close(seen[1],a.lambda_reg)
    torch.testing.assert_close(b.lambda_reg,s.lambda_reg+2)


def test_max_iter_and_face_limit(dtype,monkeypatch):
    o=oracle(dtype); s=initialize_yand_state(o,point(dtype,.1),options=opts(max_iter=1))
    install_zero(monkeypatch,LS.ZERO_STEP)
    r=yand_iteration(o,s)
    assert r.state.status==S.MAX_ITER
    o=oracle(dtype,linear=True); s=initialize_yand_state(o,point(dtype,1),options=opts(max_face_updates=1))
    r=yand_iteration(o,s)
    assert r.state.status==S.STALLED and r.state.reason==R.FACE_UPDATE_LIMIT and r.state.face_updates==1


def test_activation_invalid_keeps_accepted_x(dtype,monkeypatch):
    o=oracle(dtype,linear=True); s=initialize_yand_state(o,point(dtype,.5),options=opts())
    original=module.update_active_set
    def invalid(*args,**kw):
        r=original(*args,**kw)
        return replace(r,valid=torch.zeros_like(r.valid)) if kw['phase']=='activation' else r
    monkeypatch.setattr(module,'update_active_set',invalid)
    r=yand_iteration(o,s)
    assert r.state.status==S.INVALID_STATE and r.state.reason==R.ACTIVE_SET_FAILURE
    assert r.state.accepted_steps==1 and r.state.x is not s.x
    assert torch.equal(r.state.free_mask,s.free_mask) and r.state.face_updates==0


def test_reproducible_probes_and_release_no_draw(dtype):
    o=oracle(dtype,linear=True,n=3); options=YANDOptions(seed=876)
    x=torch.tensor([.3,.3,.4],dtype=dtype)
    a=initialize_yand_state(o,x.clone(),options=options); b=initialize_yand_state(o,x.clone(),options=options)
    r1=yand_iteration(o,a); r2=yand_iteration(o,b)
    assert torch.equal(r1.state.generator.get_state(),r2.state.generator.get_state())
    torch.testing.assert_close(r1.state.x,r2.state.x,rtol=0,atol=0)
    assert r1.state.probe_draws==1
    vertex=initialize_yand_state(o,torch.tensor([1,0,0],dtype=dtype),options=options)
    before=vertex.generator.get_state().clone()
    r=yand_iteration(o,vertex)
    assert r.event==E.RELEASE and r.state.probe_draws==0 and torch.equal(before,r.state.generator.get_state())


def test_mixed_states_and_terminal_freezing(dtype,monkeypatch):
    o=oracle(dtype)
    # optimum; vertex requiring release; positive step; injected failure.
    x=torch.tensor([[.25,.75],[0,1],[.1,.9],[.4,.6]],dtype=dtype)
    mask=torch.tensor([[True,True],[False,True],[True,True],[True,True]])
    s=initialize_yand_state(o,x,mask,options=opts())
    original=module.affine_normal_direction
    def partial(*args,**kw):
        r=original(*args,**kw); success=r.success.clone(); success[3]=False
        return replace(r,success=success)
    monkeypatch.setattr(module,'affine_normal_direction',partial)
    r=yand_iteration(o,s); new=r.state
    assert new.status.tolist()==[S.CONVERGED_KKT,S.RUNNING,S.CONVERGED_KKT,S.DIRECTION_FAILED]
    assert new.iterations.tolist()==[0,1,1,1] and new.accepted_steps.tolist()==[0,0,1,0]
    assert new.probe_draws.tolist()==[0,0,0,0]
    # Check all per-item tensor fields on a subsequent call.
    r2=yand_iteration(o,new)
    terminal=~new.active
    for name,value in vars(new).items():
        if isinstance(value,torch.Tensor):
            torch.testing.assert_close(getattr(r2.state,name)[terminal],value[terminal],rtol=0,atol=0,equal_nan=True)
    torch.testing.assert_close(r2.state.cache.z[terminal],new.cache.z[terminal],rtol=0,atol=0)


def test_batch_single_and_leading(dtype):
    o=oracle(dtype)
    x=torch.tensor([[[.1,.9],[.4,.6]],[[.25,.75],[0,1]]],dtype=dtype)
    mask=torch.tensor([[[True,True],[True,True]],[[True,True],[False,True]]])
    s=initialize_yand_state(o,x,mask,options=opts()); r=yand_iteration(o,s)
    assert r.state.x.shape==x.shape
    for i in range(2):
        for j in range(2):
            single=yand_iteration(o,initialize_yand_state(o,x[i,j],mask[i,j],options=opts())).state
            torch.testing.assert_close(single.x,r.state.x[i,j])
            assert single.status==r.state.status[i,j] and single.iterations==r.state.iterations[i,j]


def test_initial_overflow_isolation(dtype):
    scale=1e20 if dtype==torch.float32 else 1e100
    A=torch.tensor([[-scale,0],[scale,0]],dtype=dtype)
    o=MVSKOracle(MVSKData(A,torch.zeros(2,dtype=dtype),2,2),[0,0,0,1])
    s=initialize_yand_state(o,torch.tensor([[0,1],[.5,.5]],dtype=dtype),torch.ones(2,dtype=torch.bool),options=opts())
    assert s.status.tolist()==[S.CONVERGED_KKT,S.INVALID_STATE]
    assert s.reason[1]==R.NAN_DETECTED
    r=yand_iteration(o,s)
    assert r.state is s


def test_geometry_overflow_isolates_one_running_item(dtype):
    c2=1e22 if dtype==torch.float32 else 1e200
    tiny=1e-12 if dtype==torch.float32 else 1e-100
    A=torch.tensor([[-1,0],[1,0]],dtype=dtype)
    o=MVSKOracle(MVSKData(A,torch.zeros(2,dtype=dtype),2,2),[0,c2,0,0])
    x=torch.tensor([[tiny,1-tiny],[.1,.9]],dtype=dtype)
    s=initialize_yand_state(o,x,torch.ones(2,dtype=torch.bool),options=opts())
    assert s.active.all() # Both current values and raw gradients are finite.
    r=yand_iteration(o,s)
    assert r.state.status.tolist()==[S.RUNNING,S.DIRECTION_FAILED]
    assert r.state.no_progress_count[0]==1
    assert r.state.reason[1]==R.DIRECTION_NUMERICAL_FAILURE


@pytest.mark.parametrize('mode',['exact','stochastic'])
def test_real_four_asset_logdet_pipeline(dtype,mode):
    gen=torch.Generator().manual_seed(7654)
    data=prepare_mvsk_data(torch.randn(20,4,dtype=dtype,generator=gen),center_dtype=dtype)
    o=MVSKOracle(data,[.1,1,.02,.01])
    x=torch.tensor([[.25,.25,.25,.25],[.1,.2,.3,.4]],dtype=dtype)
    s=initialize_yand_state(o,x,options=YANDOptions(logdet_mode=mode,seed=345))
    assert s.active.all()
    r=yand_iteration(o,s)
    assert ((r.state.status==S.RUNNING) | (r.state.status==S.CONVERGED_KKT)).all()
    assert (r.state.accepted_steps==1).all()
    assert r.direction_result is not None and r.direction_result.logdet_status is not None
    assert r.line_search_result.success.all() and r.final_kkt.valid.all()
    assert torch.isfinite(r.state.value).all() and torch.isfinite(r.state.grad).all()
    torch.testing.assert_close(r.state.cache.z,r.state.x@o.A.T)
    assert r.state.total_pcg_iterations is None


@pytest.mark.parametrize('kind',['invalid_face','max_zero','release_no_change'])
def test_other_paths(dtype,kind,monkeypatch):
    o=oracle(dtype,linear=True)
    if kind=='invalid_face':
        s=initialize_yand_state(o,torch.tensor([.4,.4],dtype=dtype),options=opts())
        assert s.status==S.INVALID_STATE and s.reason==R.INVALID_FEASIBILITY
    elif kind=='max_zero':
        s=initialize_yand_state(o,point(dtype,.5),options=opts(max_iter=0,max_face_updates=400))
        r=yand_iteration(o,s); assert r.state.status==S.MAX_ITER and r.state.iterations==0
    else:
        original=module.update_active_set
        def unchanged(*args,**kw):
            r=original(*args,**kw); return replace(r,changed=torch.zeros_like(r.changed))
        monkeypatch.setattr(module,'update_active_set',unchanged)
        s=initialize_yand_state(o,point(dtype,1),options=opts())
        r=yand_iteration(o,s); assert r.state.no_progress_count==1 and r.state.face_updates==0


@pytest.mark.parametrize('kwargs',[{'max_iter':-1},{'stall_patience':0},{'q':0},{'lambda_growth':1},{'logdet_mode':'bad'},{'kkt_atol':-1}])
def test_options_invalid(kwargs):
    with pytest.raises(ValueError): YANDOptions(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda(dtype):
    o=oracle(dtype); x=point(dtype,.1)
    gpu=MVSKOracle(MVSKData(o.A.cuda(),o.mu.cuda(),o.T,o.n),o.coefficients.cuda())
    a=yand_iteration(o,initialize_yand_state(o,x,options=opts())).state
    b=yand_iteration(gpu,initialize_yand_state(gpu,x.cuda(),options=opts())).state
    eps=torch.finfo(dtype).eps
    torch.testing.assert_close(a.x,b.x.cpu(),rtol=512*eps,atol=512*eps)
    assert a.status==b.status.cpu() and a.iterations==b.iterations.cpu()
