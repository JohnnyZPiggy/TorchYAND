import pytest
import torch
from torchyand.diagnostics import simplex_kkt_residual


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


@pytest.mark.parametrize('x,g,mask,face,full,release', [
    ([.2,.3,.5],[2,2,2],[True]*3,True,True,[False]*3),
    ([.4,.6,0],[1,1,2],[True,True,False],True,True,[False]*3),
    ([.4,.6,0],[1,1,0],[True,True,False],True,False,[False,False,True]),
    ([1,0,0],[2,0,1],[True,False,False],True,False,[False,True,True]),
    ([1,0,0],[0,1,2],[True,False,False],True,True,[False]*3),
    ([.2,.3,.5],[0,1,2],[True]*3,False,False,[False]*3),
])
def test_reference(dtype,x,g,mask,face,full,release):
    x,g = torch.tensor(x,dtype=dtype),torch.tensor(g,dtype=dtype)
    mask=torch.tensor(mask)
    r=simplex_kkt_residual(x,g,free_mask=mask)
    free=[float(g[i]) for i in range(3) if mask[i]]
    gamma=sum(free)/len(free)
    rf=max(abs(v-gamma) for v in free)
    ra=max([0]+[max(0,gamma-float(g[i])) for i in range(3) if not mask[i]])
    torch.testing.assert_close(r.simplex_multiplier,torch.tensor(gamma,dtype=dtype))
    torch.testing.assert_close(r.face_residual,torch.tensor(rf,dtype=dtype))
    torch.testing.assert_close(r.active_violation,torch.tensor(ra,dtype=dtype))
    assert r.valid and bool(r.face_stationary)==face and bool(r.kkt_stationary)==full
    assert r.release_mask.tolist()==release
    # Lagrangian sign convention: g+lambda-eta=0.
    eta=g-gamma
    torch.testing.assert_close(g-r.simplex_multiplier-eta,torch.zeros_like(g))


@pytest.mark.parametrize('x,mask', [([1,0],[False,False]),([.5,.5],[True,False]),([1.1,-.1],[True,False]),([.4,.4],[True,True])])
def test_invalid_face(dtype,x,mask):
    r=simplex_kkt_residual(torch.tensor(x,dtype=dtype),torch.zeros(2,dtype=dtype),free_mask=torch.tensor(mask))
    assert not r.valid and not r.face_stationary and not r.kkt_stationary
    if not any(mask):
        assert torch.isnan(r.simplex_multiplier)


def test_batch_broadcast_and_single(dtype):
    x=torch.tensor([[.5,.5,0],[1,0,0],[.2,.3,.5]],dtype=dtype)
    g=torch.tensor([[1,1,0],[0,1,2],[0,1,2]],dtype=dtype)
    r=simplex_kkt_residual(x,g)
    assert r.face_stationary.tolist()==[True,True,False]
    assert r.kkt_stationary.tolist()==[False,True,False]
    for i in range(3):
        s=simplex_kkt_residual(x[i],g[i])
        for name in vars(r):
            torch.testing.assert_close(getattr(r,name)[i],getattr(s,name))
    b=simplex_kkt_residual(x[2],g,free_mask=torch.ones(3,dtype=torch.bool))
    assert b.release_mask.shape==(3,3)


def test_tolerance_and_lower_bound(dtype):
    eps=torch.finfo(dtype).eps
    x=torch.tensor([.4,.5,.1],dtype=dtype)
    g=torch.tensor([2,2,3],dtype=dtype)
    r=simplex_kkt_residual(x,g,tau=.1)
    assert r.kkt_stationary
    default=1e-5 if dtype==torch.float32 else 1e-10
    torch.testing.assert_close(r.stopping_threshold,torch.tensor(3*default,dtype=dtype))
    r=simplex_kkt_residual(torch.tensor([1+eps,-eps],dtype=dtype),torch.zeros(2,dtype=dtype))
    assert r.feasible and r.valid


def test_explicit_tolerance_and_singleton(dtype):
    r=simplex_kkt_residual(torch.tensor([1.],dtype=dtype),torch.tensor([99.],dtype=dtype))
    assert r.valid and r.kkt_stationary and r.face_residual==0
    x=torch.tensor([.5,.5],dtype=dtype); g=torch.tensor([1.,1.002],dtype=dtype)
    strict=simplex_kkt_residual(x,g,rtol=0,atol=0)
    loose=simplex_kkt_residual(x,g,rtol=0,atol=.01,feasibility_tol=1e-6)
    assert not strict.face_stationary and loose.kkt_stationary


@pytest.mark.parametrize('tau', [0.0, 0.1])
def test_released_at_bound_validity(dtype,tau):
    eps=torch.finfo(dtype).eps
    x=torch.tensor([tau,.4,.6-tau],dtype=dtype)
    g=torch.tensor([0,1,1],dtype=dtype)
    # Explicit all-free mask permits a coordinate still at the lower bound.
    all_free=torch.ones(3,dtype=torch.bool)
    r=simplex_kkt_residual(x,g,tau=tau,free_mask=all_free)
    assert r.valid and r.feasible and not r.face_stationary
    # An active coordinate away from the bound is still inconsistent.
    bad=simplex_kkt_residual(x,g,tau=tau,free_mask=torch.tensor([True,False,True]))
    assert bad.feasible and not bad.valid
    # A valid active coordinate at tau can be released without changing x.
    mask=torch.tensor([False,True,True])
    before=simplex_kkt_residual(x,g,tau=tau,free_mask=mask)
    assert before.valid and before.face_stationary and not before.kkt_stationary
    assert before.release_mask.tolist()==[True,False,False]
    new_mask=mask.clone(); new_mask[0]=True
    after=simplex_kkt_residual(x,g,tau=tau,free_mask=new_mask)
    assert after.valid and after.free_count==3
    torch.testing.assert_close(after.simplex_multiplier,torch.tensor(2/3,dtype=dtype))
    torch.testing.assert_close(after.face_residual,torch.tensor(2/3,dtype=dtype))
    assert not after.release_mask.any()
    # Free coordinates above, at, and just below the bound are permitted
    # within bound_tol. Beyond tolerance feasibility and validity fail.
    shifts=torch.tensor([eps,0,-eps,-128*eps],dtype=dtype)
    batch=x.expand(4,3).clone()
    batch[:,0]+=shifts; batch[:,2]-=shifts
    result=simplex_kkt_residual(batch,g,tau=tau,free_mask=all_free)
    assert result.valid.tolist()==[True,True,True,False]


def test_explicit_bound_tolerance_broadcast(dtype):
    x=torch.tensor([[-1e-4,.4,.6001],[-1e-4,.4,.6001]],dtype=dtype)
    mask=torch.tensor([True,True,True])
    result=simplex_kkt_residual(x,torch.zeros_like(x),free_mask=mask,
                               feasibility_tol=torch.tensor([2e-4,5e-5],dtype=dtype))
    assert result.valid.tolist()==[True,False]
    # Reclassifying the boundary-near coordinate as active uses the same tol.
    mask=torch.tensor([False,True,True])
    result=simplex_kkt_residual(x,torch.zeros_like(x),free_mask=mask,
                               feasibility_tol=torch.tensor([2e-4,5e-5],dtype=dtype))
    assert result.valid.tolist()==[True,False]


@pytest.mark.parametrize('case',['dtype','asset','nan','mask','tol'])
def test_validation(case):
    x=torch.tensor([.5,.5],dtype=torch.float64); g=x.clone()
    kwargs={}
    if case=='dtype': g=g.float()
    if case=='asset': g=g[:1]
    if case=='nan': g[0]=float('nan')
    if case=='mask': kwargs['free_mask']=torch.ones(1,dtype=torch.bool)
    if case=='tol': kwargs['rtol']=-1
    with pytest.raises((ValueError,TypeError)):
        simplex_kkt_residual(x,g,**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda(dtype):
    x=torch.tensor([[.4,.6,0],[1,0,0]],dtype=dtype); g=torch.tensor([[1,1,0],[0,1,2]],dtype=dtype)
    a=simplex_kkt_residual(x,g); b=simplex_kkt_residual(x.cuda(),g.cuda())
    for name in vars(a): torch.testing.assert_close(getattr(a,name),getattr(b,name).cpu())
