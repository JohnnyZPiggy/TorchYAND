"""Feasible line-search integration, independent roots/grid and failure masks."""
from dataclasses import replace
import math
import numpy as np
import pytest
import torch
from torchyand.data import MVSKData
from torchyand.oracle import MVSKOracle
from torchyand.line_search import exact_mvsk_line_search, LineSearchStatus as S
from torchyand.line_search import mvsk, kernels


@pytest.fixture(params=[torch.float32,torch.float64])
def dtype(request): return request.param


def model(dtype,weights=(1,1,0,1),mean=.75,asymmetric=False):
    k=[-1.,-1.,2.] if asymmetric else [-1.,1.]
    A=torch.tensor([[v,0] for v in k],dtype=dtype)
    return MVSKOracle(MVSKData(A,torch.tensor([mean,0],dtype=dtype),len(k),2),weights)


def inputs(dtype,t=.1,s=.1):
    return torch.tensor([t,1-t],dtype=dtype),torch.tensor([s,-s],dtype=dtype)


def reference(o,x,q,cap):
    A=o.A.double().numpy(); mu=o.mu.double().numpy(); c=o.coefficients.double().numpy()
    z=A@x.double().numpy(); w=A@q.double().numpy()
    coeff=np.zeros(5)
    coeff[0]=-c[0]*mu@x.double().numpy(); coeff[1]=-c[0]*mu@q.double().numpy()
    for degree,weight in ((2,c[1]),(3,-c[2]),(4,c[3])):
        for j in range(degree+1): coeff[j]+=weight*math.comb(degree,j)*np.mean(z**(degree-j)*w**j)
    der=np.array([j*coeff[j] for j in range(1,5)])
    roots=np.polynomial.polynomial.polyroots(np.trim_zeros(der,'b')) if np.any(der) else []
    candidates=[0.,cap]+[float(r.real) for r in roots if abs(r.imag)<=1e-8*max(1,abs(r.real)) and 0<=r.real<=cap]
    values=np.polynomial.polynomial.polyval(candidates,coeff)
    best=int(np.argmin(values))
    return candidates[best],values[best],coeff


@pytest.mark.parametrize('weights,mean,asym', [((1,1,0,1),.75,False),((1,1,0,1),20,False),
    ((1,1,0,1),-1,False),((1,0,0,0),1,False),((1,1,0,0),.75,False),
    ((1,1,.2,0),.75,True),((0,0,0,0),0,False)])
def test_independent_roots_grid(dtype,weights,mean,asym):
    o=model(dtype,weights,mean,asym); x,d=inputs(dtype)
    r=exact_mvsk_line_search(o,x,d)
    assert r.success
    alpha,value,coeff=reference(o,x,r.direction,float(r.alpha_max))
    eps=torch.finfo(dtype).eps
    tol=1024*eps*max(1,abs(value),np.abs(coeff).max()*max(1,float(r.alpha_max))**4)
    assert abs(float(r.trial_value)-value)<=tol
    assert 0<=r.alpha<=r.alpha_max
    assert r.x_trial.min()>=-64*eps
    assert abs(float(r.x_trial.sum())-1)<=64*eps
    torch.testing.assert_close(r.trial_value,o.value(r.x_trial))
    assert abs(float(r.predicted_value-r.trial_value))<=tol
    assert r.trial_value<=r.current_value+128*eps*max(1,abs(float(r.current_value)),abs(float(r.trial_value)))
    grid=np.linspace(0,float(r.alpha_max),2001)
    assert float(r.trial_value)<=np.polynomial.polynomial.polyval(grid,coeff).min()+tol
    if mean==20 or weights==(1,0,0,0):
        assert r.hits_boundary
        assert torch.equal(r.hit_mask,r.step_cap_result.hit_mask)
    if mean<0 or weights==(0,0,0,0): assert r.status==S.ZERO_STEP and r.alpha==0
    if mean==.75 and weights==(1,1,0,1): assert r.alpha>1 and not r.hits_boundary


def test_projected_direction_and_cache(dtype):
    o=model(dtype); x,d=inputs(dtype)
    d=d+2 # q must remove the normal simplex component.
    r=exact_mvsk_line_search(o,x,d,cache=o.cache(x))
    a=exact_mvsk_line_search(o,x,d)
    torch.testing.assert_close(r.direction,torch.tensor([.1,-.1],dtype=dtype))
    torch.testing.assert_close(r.x_trial,x+r.alpha*r.direction)
    torch.testing.assert_close(r.alpha,a.alpha,rtol=0,atol=0)
    assert r.success
    with pytest.raises(ValueError): exact_mvsk_line_search(o,x.clone(),d,cache=o.cache(x))


def test_hand_computed_and_active_face(dtype):
    # f(t)=t²-.5t, x0=.1, q0=.1 => alpha*=1.5, f*=-1/16.
    o=model(dtype,(1,1,0,0),.5); x,d=inputs(dtype)
    r=exact_mvsk_line_search(o,x,d)
    eps=torch.finfo(dtype).eps
    torch.testing.assert_close(r.alpha,torch.tensor(1.5,dtype=dtype),rtol=128*eps,atol=128*eps)
    torch.testing.assert_close(r.trial_value,torch.tensor(-1/16,dtype=dtype),rtol=128*eps,atol=128*eps)
    data=MVSKData(torch.cat((o.A,torch.zeros(o.T,1,dtype=dtype)),-1),torch.tensor([.5,0,0],dtype=dtype),o.T,3)
    o=MVSKOracle(data,(1,1,0,0))
    r=exact_mvsk_line_search(o,torch.tensor([.1,.9,0],dtype=dtype),torch.tensor([.1,-.1,999],dtype=dtype),free_mask=torch.tensor([True,True,False]))
    assert r.success and r.direction[2]==0 and r.x_trial[2]==0 and not r.hit_mask[2]


def test_mixed_downstream_failure_masks(dtype,monkeypatch):
    o=model(dtype); x,d=inputs(dtype)
    x=x.expand(4,2); d=d.expand(4,2)
    original_assembly=kernels.mvsk_quartic_coefficients_unchecked
    def assembly(*args,**kwargs):
        c,finite=original_assembly(*args,**kwargs)
        finite=finite.clone(); finite[1]=False
        return c,finite
    monkeypatch.setattr(kernels,'mvsk_quartic_coefficients_unchecked',assembly)
    original_quartic=kernels._minimize_quartic_interval_unchecked
    def quartic(*args,**kwargs):
        r=original_quartic(*args,**kwargs)
        s=r.success.clone(); s[2]=False
        return replace(r,success=s)
    monkeypatch.setattr(kernels,'_minimize_quartic_interval_unchecked',quartic)
    original_value=kernels.value; calls=[0]
    def value(*args,**kwargs):
        calls[0]+=1
        v=original_value(*args,**kwargs)
        if calls[0]==2:
            v=v.clone(); v[3]+=10
        return v
    monkeypatch.setattr(kernels,'value',value)
    r=exact_mvsk_line_search(o,x,d)
    assert r.status.tolist()==[S.SUCCESS,S.ASSEMBLY_FAILED,S.QUARTIC_FAILED,S.VERIFICATION_FAILED]
    assert r.success.tolist()==[True,False,False,False]
    assert (r.alpha[1:]==0).all()
    torch.testing.assert_close(r.x_trial[1:],x[1:])
    # Dummy minimizer success must not erase the assembly failure.
    assert r.quartic_result.success[1]


@pytest.mark.parametrize('kind',['direction','cap'])
def test_zero_special_skips_quartic(dtype,kind,monkeypatch):
    o=model(dtype); x,d=inputs(dtype)
    if kind=='direction': d=torch.zeros_like(d)
    else: x,d=inputs(dtype,t=0,s=-.1)
    def forbidden(*args,**kwargs): raise AssertionError('must skip quartic')
    monkeypatch.setattr(kernels,'_minimize_quartic_interval_unchecked',forbidden)
    r=exact_mvsk_line_search(o,x,d,free_mask=torch.ones(2,dtype=torch.bool))
    assert r.success and r.alpha==0 and not r.hits_boundary and r.quartic_result is None
    assert r.status==(S.ZERO_DIRECTION if kind=='direction' else S.ZERO_CAP)
    torch.testing.assert_close(r.x_trial,x,rtol=0,atol=0)


def test_tied_minimum(dtype):
    # f(t)=t²(t-1)²: sample second/third/fourth moments 2,2,6.
    o=model(dtype,(0,.5,1,1/6),0,True); x,d=inputs(dtype,t=0,s=.2)
    r=exact_mvsk_line_search(o,x,d,free_mask=torch.ones(2,dtype=torch.bool))
    assert r.success and r.status==S.ZERO_STEP and r.alpha==0 and not r.hits_boundary


def test_mixed_batch(dtype):
    o=model(dtype)
    x=torch.tensor([[.1,.9],[.1,.9],[0,1],[.9,.1],[.1,.9]],dtype=dtype)
    d=torch.tensor([[.1,-.1],[0,0],[-.1,.1],[.1,-.1],[1e-12,-1e-12]],dtype=dtype)
    # Explicit huge direction tolerance makes only the tiny item unreliable.
    r=exact_mvsk_line_search(o,x,d,free_mask=torch.ones(2,dtype=torch.bool),direction_tol=1e-10)
    assert r.status.tolist()==[S.SUCCESS,S.ZERO_DIRECTION,S.ZERO_CAP,S.ZERO_STEP,S.STEP_CAP_FAILED]
    assert r.success.tolist()==[True,True,True,True,False]
    assert r.alpha[-1]==0
    torch.testing.assert_close(r.x_trial[-1],x[-1])
    for i in range(5):
        s=exact_mvsk_line_search(o,x[i],d[i],free_mask=torch.ones(2,dtype=torch.bool),direction_tol=1e-10)
        torch.testing.assert_close(s.alpha,r.alpha[i])
        assert s.status==r.status[i]
    b=exact_mvsk_line_search(o,x[:2,None,:],d[:3],free_mask=torch.ones(2,dtype=torch.bool))
    assert b.x_trial.shape==(2,3,2)


def test_assembly_overflow_mixed(dtype):
    o=model(dtype); x,_=inputs(dtype)
    huge=1e12 if dtype==torch.float32 else 1e100
    d=torch.tensor([[.1,-.1],[huge,-huge]],dtype=dtype)
    r=exact_mvsk_line_search(o,x,d)
    assert r.success.tolist()==[True,False]
    assert r.status[1]==S.ASSEMBLY_FAILED
    assert r.alpha[1]==0
    torch.testing.assert_close(r.x_trial[1],x)


@pytest.mark.parametrize('failure',['quartic','outside','infeasible','true_value','predicted','nonfinite_cap'])
def test_failure_propagation(dtype,failure,monkeypatch):
    o=model(dtype); x,d=inputs(dtype)
    if failure in ('quartic','outside'):
        original=kernels._minimize_quartic_interval_unchecked
        def altered(*args,**kwargs):
            r=original(*args,**kwargs)
            return replace(r,success=torch.zeros_like(r.success)) if failure=='quartic' else replace(r,x=torch.full_like(r.x,100))
        monkeypatch.setattr(kernels,'_minimize_quartic_interval_unchecked',altered)
        expected=S.QUARTIC_FAILED
    elif failure=='infeasible':
        # Simulate a downstream feasibility verification fault, not repair.
        original=kernels.face_validity
        def altered(*args,**kwargs):
            f,v=original(*args,**kwargs); return torch.zeros_like(f),v
        monkeypatch.setattr(kernels,'face_validity',altered); expected=S.TRIAL_INFEASIBLE
    elif failure=='true_value':
        original=kernels.value; calls=[0]
        def altered(*args,**kwargs):
            calls[0]+=1
            v=original(*args,**kwargs)
            return v+10 if calls[0]==2 else v
        monkeypatch.setattr(kernels,'value',altered); expected=S.VERIFICATION_FAILED
    elif failure=='predicted':
        original=kernels.mvsk_quartic_coefficients_unchecked
        def altered(*args,**kwargs):
            c,f=original(*args,**kwargs); c=c.clone(); c[...,0]-=10; return c,f
        monkeypatch.setattr(kernels,'mvsk_quartic_coefficients_unchecked',altered); expected=S.VERIFICATION_FAILED
    else:
        original=mvsk.simplex_step_cap
        def altered(*args,**kwargs):
            r=original(*args,**kwargs); return replace(r,alpha_max=torch.full_like(r.alpha_max,float('inf')))
        monkeypatch.setattr(mvsk,'simplex_step_cap',altered); expected=S.NONFINITE_CAP
    r=exact_mvsk_line_search(o,x,d)
    assert not r.success and r.status==expected and r.alpha==0 and not r.hits_boundary
    torch.testing.assert_close(r.x_trial,x,rtol=0,atol=0)


def test_huge_a0_and_cancellation(dtype):
    o=model(dtype)
    # Common mean offset contributes a huge A0 but zero directional mean.
    o=MVSKOracle(MVSKData(o.A,torch.tensor([1e10,1e10],dtype=dtype),o.T,o.n),[1,1,0,1])
    x,d=inputs(dtype,t=.8,s=-.1)
    r=exact_mvsk_line_search(o,x,d)
    assert r.success and r.alpha>0
    assert torch.isfinite(r.predicted_value) and torch.isfinite(r.trial_value)


@pytest.mark.parametrize('case',['dtype','asset','nan','tol','owner'])
def test_validation(case):
    o=model(torch.float64); x,d=inputs(torch.float64); kwargs={}
    if case=='dtype': d=d.float()
    if case=='asset': d=d[:1]
    if case=='nan': d[0]=float('nan')
    if case=='tol': kwargs['acceptance_tol']=-1
    if case=='owner': kwargs['cache']=replace(o.cache(x),owner=object())
    with pytest.raises((TypeError,ValueError)):
        exact_mvsk_line_search(o,x,d,**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda(dtype):
    o=model(dtype); x,d=inputs(dtype)
    gpu=MVSKOracle(MVSKData(o.A.cuda(),o.mu.cuda(),o.T,o.n),o.coefficients.cuda())
    a=exact_mvsk_line_search(o,x,d); b=exact_mvsk_line_search(gpu,x.cuda(),d.cuda())
    eps=torch.finfo(dtype).eps
    torch.testing.assert_close(a.alpha,b.alpha.cpu(),rtol=512*eps,atol=512*eps)
    assert a.status==b.status.cpu() and torch.equal(a.hit_mask,b.hit_mask.cpu())
