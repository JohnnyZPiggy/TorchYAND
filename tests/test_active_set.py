import pytest
import torch
from torchyand.constraints.active_set import update_active_set, ActiveSetStatus as S
from torchyand.diagnostics import simplex_kkt_residual


@pytest.fixture(params=[torch.float32,torch.float64])
def dtype(request): return request.param


def call(dtype,x,g,mask,phase,**kw):
    return update_active_set(torch.tensor(x,dtype=dtype),torch.tensor(g,dtype=dtype),torch.tensor(mask),phase=phase,**kw)


@pytest.mark.parametrize('x,mask,requested,expected,status',[
    ([0,.4,.6],[True]*3,[True,False,False],[False,True,True],S.ACTIVATED),
    ([0,0,1],[True]*3,[True,True,False],[False,False,True],S.ACTIVATED),
    ([0,.4,.6],[False,True,True],[True,False,False],[False,True,True],S.INVALID_REQUEST),
    ([.2,.3,.5],[True]*3,[True,False,False],[True]*3,S.INVALID_REQUEST),
    ([.2,.3,.5],[True]*3,[False]*3,[True]*3,S.NO_CHANGE),
])
def test_activation(dtype,x,mask,requested,expected,status):
    x=torch.tensor(x,dtype=dtype); saved=x.clone(); mask=torch.tensor(mask); saved_mask=mask.clone()
    r=update_active_set(x,torch.zeros_like(x),mask,phase='activation',activation_mask=torch.tensor(requested))
    assert r.status==status and r.free_mask.tolist()==expected
    assert bool(r.valid)==(status!=S.INVALID_REQUEST)
    assert r.free_mask.any() and not r.released_mask.any()
    assert torch.equal(x,saved) and torch.equal(mask,saved_mask)
    assert torch.equal(r.activated_mask,mask & ~r.free_mask)


@pytest.mark.parametrize('kind',['last','max','tie'])
def test_preservation(dtype,kind):
    if kind=='last':
        x=[1,0,0]; mask=[True,False,False]; request=[True,False,False]; tau=0.; tol=1.; keep=0
    else:
        # Singleton lower-bound slice meets ordinary validity; no special
        # interpretation is added. Small roundoff-scale differences suffice.
        eps=torch.finfo(dtype).eps
        x=[1/3,1/3,1/3]; mask=[True]*3; request=[True]*3; tau=1/3; tol=64*eps; keep=0
        if kind=='max': x=[1/3-eps,1/3+eps,1/3]; keep=1
    r=call(dtype,x,[0]*3,mask,'activation',activation_mask=torch.tensor(request),tau=tau,active_tol=tol)
    assert r.valid and r.status==S.LAST_FREE_PRESERVED
    assert r.free_mask.tolist()==[i==keep for i in range(3)]
    assert bool(r.changed)==(kind!='last')


@pytest.mark.parametrize('x,g,mask,selected',[
    ([0,.4,.6],[0,1,1],[False,True,True],0),
    ([0,0,1],[0,-1,2],[False,False,True],1),
    ([0,0,1],[0,0,2],[False,False,True],0),
    ([0,.4,.6],[2,1,1],[False,True,True],None),
    ([.2,.3,.5],[0,1,2],[True]*3,None),
    ([0,.4,.6],[0,1,2],[False,True,True],None),
])
def test_release(dtype,x,g,mask,selected):
    r=call(dtype,x,g,mask,'release')
    assert r.valid
    assert r.status==(S.NO_CHANGE if selected is None else S.RELEASED)
    assert r.released_mask.sum()==(0 if selected is None else 1)
    if selected is not None:
        assert r.released_mask[selected]
        new=simplex_kkt_residual(torch.tensor(x,dtype=dtype),torch.tensor(g,dtype=dtype),free_mask=r.free_mask)
        assert new.valid and x[selected]==0


@pytest.mark.parametrize('requested,status,chosen', [([True,False,False],S.RELEASED,0),([False,True,False],S.RELEASED,1),([False,False,False],S.NO_CHANGE,None),([False,False,True],S.INVALID_REQUEST,None)])
def test_release_request(dtype,requested,status,chosen):
    r=call(dtype,[0,0,1],[0,-1,2],[False,False,True],'release',release_mask=torch.tensor(requested))
    assert r.status==status
    assert bool(r.valid)==(status!=S.INVALID_REQUEST)
    assert r.released_mask.sum()==(0 if chosen is None else 1)
    if chosen is not None: assert r.released_mask[chosen]


@pytest.mark.parametrize('phase',['activation','release'])
@pytest.mark.parametrize('x,mask', [([1,0],[False,False]),([.5,.5],[True,False]),([.4,.4],[True,True]),([1.1,-.1],[True,True])])
def test_invalid_face(dtype,phase,x,mask):
    r=call(dtype,x,[0,0],mask,phase)
    assert not r.valid and r.status==S.INVALID_FACE and not r.changed
    assert r.free_mask.tolist()==mask and not r.activated_mask.any() and not r.released_mask.any()


def test_workflow(dtype):
    x=torch.tensor([0,.4,.6],dtype=dtype); g=torch.tensor([0,1,1],dtype=dtype)
    mask=torch.ones(3,dtype=torch.bool); original=x.clone()
    a=update_active_set(x,g,mask,phase='activation',activation_mask=torch.tensor([True,False,False]))
    b=update_active_set(x,g,a.free_mask,phase='release')
    assert a.status==S.ACTIVATED and b.status==S.RELEASED
    assert torch.equal(b.free_mask,mask) and torch.equal(x,original)
    assert simplex_kkt_residual(x,g,free_mask=b.free_mask).valid


@pytest.mark.parametrize('phase',['activation','release'])
def test_batch_single_and_broadcast(dtype,phase):
    x=torch.tensor([[0,.4,.6],[0,.4,.6],[0,0,1],[.4,.4,0]],dtype=dtype)
    g=torch.tensor([[0,1,1],[2,1,1],[0,-1,2],[0,0,0]],dtype=dtype)
    mask=torch.tensor([[False,True,True],[False,True,True],[False,False,True],[True,True,False]])
    request=torch.tensor([[False,False,False],[True,False,False],[False,False,False],[False,False,False]])
    kwargs={'activation_mask':request} if phase=='activation' else {}
    r=update_active_set(x,g,mask,phase=phase,**kwargs)
    expected=[S.NO_CHANGE,S.INVALID_REQUEST,S.NO_CHANGE,S.INVALID_FACE] if phase=='activation' else [S.RELEASED,S.NO_CHANGE,S.RELEASED,S.INVALID_FACE]
    assert r.status.tolist()==expected
    for i in range(4):
        kw={'activation_mask':request[i]} if phase=='activation' else {}
        s=update_active_set(x[i],g[i],mask[i],phase=phase,**kw)
        for name in vars(r): torch.testing.assert_close(getattr(r,name)[i],getattr(s,name))
    b=update_active_set(x[:2,None,:],g[:3],mask[:2,None,:],phase=phase)
    assert b.free_mask.shape==(2,3,3)


@pytest.mark.parametrize('case',['phase','both','dtype','asset','nonfinite','mask','tol'])
def test_validation(case):
    x=torch.tensor([.4,.6],dtype=torch.float64); g=x.clone(); mask=torch.ones(2,dtype=torch.bool)
    kwargs={'phase':'activation'}
    if case=='phase': kwargs['phase']='both'
    if case=='both': kwargs['release_mask']=mask
    if case=='dtype': g=g.float()
    if case=='asset': g=g[:1]
    if case=='nonfinite': g[0]=float('nan')
    if case=='mask': kwargs['activation_mask']=torch.ones(1,dtype=torch.bool)
    if case=='tol': kwargs['active_tol']=-1
    with pytest.raises((TypeError,ValueError)): update_active_set(x,g,mask,**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('phase',['activation','release'])
def test_cuda(dtype,phase):
    x=torch.tensor([0,.4,.6],dtype=dtype); g=torch.tensor([0,1,1],dtype=dtype)
    mask=torch.tensor([False,True,True]) if phase=='release' else torch.ones(3,dtype=torch.bool)
    request=torch.tensor([True,False,False])
    kwargs={'activation_mask':request} if phase=='activation' else {}
    a=update_active_set(x,g,mask,phase=phase,**kwargs)
    kwargs={'activation_mask':request.cuda()} if phase=='activation' else {}
    b=update_active_set(x.cuda(),g.cuda(),mask.cuda(),phase=phase,**kwargs)
    for name in vars(a): torch.testing.assert_close(getattr(a,name),getattr(b,name).cpu())
