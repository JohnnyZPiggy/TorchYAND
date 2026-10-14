import pytest
import torch
from torchyand.constraints import simplex_step_cap, StepCapStatus


@pytest.fixture(params=[torch.float32,torch.float64])
def dtype(request): return request.param


@pytest.mark.parametrize('x,d,cap,hits',[
    ([.2,.3,.5],[-1,.5,.5],.2,[True,False,False]),
    ([.2,.3,.5],[-.2,-.3,.5],1,[True,True,False]),
    ([.2,.3,.5],[-.1,0,.1],2,[True,False,False]),
    ([0,.5,.5],[-1,.5,.5],0,[True,False,False]),
])
def test_reference(dtype,x,d,cap,hits):
    x,d=torch.tensor(x,dtype=dtype),torch.tensor(d,dtype=dtype)
    r=simplex_step_cap(x,d,free_mask=torch.ones(3,dtype=torch.bool))
    assert r.success and r.hit_mask.tolist()==hits
    torch.testing.assert_close(r.alpha_max,torch.tensor(cap,dtype=dtype))
    assert (x+r.alpha_max*r.direction >= -64*torch.finfo(dtype).eps).all()
    assert (x+(r.alpha_max+.01)*r.direction < 0).any()


@pytest.mark.parametrize('d',[[0,0,0],[1,1,1]])
def test_zero_and_no_negative(dtype,d):
    r=simplex_step_cap(torch.full((3,),1/3,dtype=dtype),torch.tensor(d,dtype=dtype))
    assert r.success and r.zero_direction and torch.isinf(r.alpha_max)
    assert not r.has_boundary_hit and not r.hit_mask.any()


def test_tiny_negative(dtype):
    tiny=torch.finfo(dtype).eps/100
    r=simplex_step_cap(torch.tensor([.5,.5],dtype=dtype),torch.tensor([-tiny,tiny],dtype=dtype))
    assert not r.success and r.status==StepCapStatus.NO_RELIABLE_NEGATIVE
    assert torch.isfinite(r.alpha_max)
    torch.testing.assert_close(r.alpha_max,torch.tensor(.5/tiny,dtype=dtype))
    # A tiny negative, missed by the diagnostic threshold, is the limiter.
    x=torch.tensor([tiny/10,.5,.5-tiny/10],dtype=dtype)
    d=torch.tensor([-tiny,-1,1],dtype=dtype)
    r=simplex_step_cap(x,d,free_mask=torch.ones(3,dtype=torch.bool))
    expected=x[0]/(-r.direction[0])
    assert r.success
    torch.testing.assert_close(r.alpha_max,expected)
    assert r.alpha_max < .5 and r.hit_mask[0]


def test_active_and_projection(dtype):
    x=torch.tensor([.4,.6,0],dtype=dtype)
    r=simplex_step_cap(x,torch.tensor([-2,0,99],dtype=dtype),free_mask=torch.tensor([True,True,False]))
    torch.testing.assert_close(r.direction,torch.tensor([-1,1,0],dtype=dtype))
    assert r.success and not r.hit_mask[2]


def test_overflow(dtype):
    tiny=torch.finfo(dtype).tiny/100
    r=simplex_step_cap(torch.tensor([.5,.5],dtype=dtype),torch.tensor([-tiny,tiny],dtype=dtype),direction_tol=0)
    assert not r.success and r.status==StepCapStatus.NONFINITE_RATIO
    assert not r.zero_direction


def test_batch_and_hit_tolerance(dtype):
    x=torch.tensor([[.2,.3,.5],[.2,.3,.5],[0,.5,.5]],dtype=dtype)
    d=torch.tensor([[-1,.5,.5],[0,0,0],[-1,.5,.5]],dtype=dtype)
    mask=torch.ones(3,dtype=torch.bool)
    r=simplex_step_cap(x,d,free_mask=mask)
    assert r.success.all()
    for i in range(3):
        s=simplex_step_cap(x[i],d[i],free_mask=mask)
        for name in vars(r): torch.testing.assert_close(getattr(r,name)[i],getattr(s,name))
    r=simplex_step_cap(x,d,free_mask=mask,hit_tol=.1)
    assert r.hit_mask.shape==(3,3)


@pytest.mark.parametrize('x,mask', [([.4,.4],[True,True]),([.5,.5],[True,False]),([1.1,-.1],[True,True]),([1,0],[False,False])])
def test_invalid(dtype,x,mask):
    r=simplex_step_cap(torch.tensor(x,dtype=dtype),torch.tensor([-1,1],dtype=dtype),free_mask=torch.tensor(mask))
    assert not r.success and r.status==StepCapStatus.INVALID_INPUT_FACE


def test_lower_bound_and_leading_broadcast(dtype):
    x=torch.tensor([.4,.5,.1],dtype=dtype)
    d=torch.tensor([[[-1,1,0]],[[0,0,0]]],dtype=dtype)
    r=simplex_step_cap(x,d,tau=.1,free_mask=torch.tensor([True,True,False]))
    assert r.direction.shape==(2,1,3) and r.alpha_max.shape==(2,1)
    torch.testing.assert_close(r.alpha_max[0,0],torch.tensor(.3,dtype=dtype))
    assert r.success.all() and not r.hit_mask[...,2].any()


@pytest.mark.parametrize('case',['dtype','asset','nan','mask','tol'])
def test_validation(case):
    x=torch.tensor([.5,.5],dtype=torch.float64); d=x.clone(); kwargs={}
    if case=='dtype': d=d.float()
    if case=='asset': d=d[:1]
    if case=='nan': d[0]=float('inf')
    if case=='mask': kwargs['free_mask']=torch.ones(1,dtype=torch.bool)
    if case=='tol': kwargs['hit_tol']=-1
    with pytest.raises((TypeError,ValueError)):
        simplex_step_cap(x,d,**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda(dtype):
    x=torch.tensor([.2,.3,.5],dtype=dtype); d=torch.tensor([-1,.5,.5],dtype=dtype)
    a=simplex_step_cap(x,d); b=simplex_step_cap(x.cuda(),d.cuda())
    for name in vars(a): torch.testing.assert_close(getattr(a,name),getattr(b,name).cpu())
