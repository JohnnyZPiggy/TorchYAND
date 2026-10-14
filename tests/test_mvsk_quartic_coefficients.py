"""Independent expansion/derivative references for exact MVSK line models."""
import math
import pytest
import torch
from torchyand.data import prepare_mvsk_data, MVSKData
from torchyand.oracle import MVSKOracle
from torchyand.line_models import mvsk_quartic_coefficients


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request): return request.param


def problem(dtype, scale=1., coefficients=(.2,.3,.4,.5)):
    gen=torch.Generator().manual_seed(20261104)
    R=torch.randn(19,4,generator=gen,dtype=dtype)*scale
    oracle=MVSKOracle(prepare_mvsk_data(R,center_dtype=dtype),coefficients)
    x=torch.randn(2,3,4,generator=gen,dtype=dtype)
    d=torch.randn(2,3,4,generator=gen,dtype=dtype)
    return oracle,x,d


def expansion(oracle,x,d):
    # Independent binomial expansion, double reference of the stored data.
    A,mu,c=oracle.A.double(),oracle.mu.double(),oracle.coefficients.double()
    x,d=x.double(),d.double()
    z,w=x@A.T,d@A.T
    out=[]; contributions=[]
    for j in range(5):
        terms=[]; scales=[]
        if j==0:
            terms.append(-c[0]*(x*mu).sum(-1)); scales.append(c[0]*(x*mu).abs().sum(-1))
        if j==1:
            terms.append(-c[0]*(d*mu).sum(-1)); scales.append(c[0]*(d*mu).abs().sum(-1))
        for k,weight in ((2,c[1]),(3,-c[2]),(4,c[3])):
            if j<=k:
                contribution=weight*math.comb(k,j)*(z**(k-j)*w**j)
                terms.append(contribution.mean(-1)); scales.append(contribution.abs().mean(-1))
        out.append(sum(terms))
        contributions.append(sum(scales))
    return torch.stack(out,-1),torch.stack(contributions,-1)


def close(actual,expected,scale,dtype):
    # Cancellation-sensitive absolute budget; relative error alone is inadequate.
    budget=256*torch.finfo(dtype).eps*torch.as_tensor(scale).clamp_min(1)
    assert torch.all((actual.double()-expected.double()).abs()<=budget), (actual,expected,budget)


def test_hand_example(dtype):
    # z=[-x,x], mu=2, all c=1: f(x)=-2x+x²+x⁴.
    oracle=MVSKOracle(MVSKData(torch.tensor([[-1.],[1.]],dtype=dtype),torch.tensor([2.],dtype=dtype),2,1),[1,1,1,1])
    coeff=mvsk_quartic_coefficients(oracle,torch.tensor([1.],dtype=dtype),torch.tensor([2.],dtype=dtype))
    torch.testing.assert_close(coeff,torch.tensor([0,8,28,32,16],dtype=dtype))


@pytest.mark.parametrize('weights',[(0,0,0,0),(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1),(.2,.3,.4,.5)])
def test_expansion_and_evaluation(dtype,weights):
    o,x,d=problem(dtype,coefficients=weights)
    coeff=mvsk_quartic_coefficients(o,x,d)
    ref,scale=expansion(o,x,d)
    close(coeff,ref,scale,dtype)
    for alpha in (-1.2,-.1,0.,.3,1.5):
        p=coeff[...,4]
        for j in (3,2,1,0): p=p*alpha+coeff[...,j]
        direct=o.value(x+alpha*d)
        magnitude=sum(scale[...,j]*abs(alpha)**j for j in range(5))
        close(p,direct,magnitude,dtype)


def test_derivative_identities(dtype):
    o,x,d=problem(dtype)
    coeff=mvsk_quartic_coefficients(o,x,d)
    _,scale=expansion(o,x,d)
    torch.testing.assert_close(coeff[...,0],o.value(x),rtol=0,atol=0)
    close(coeff[...,1],(o.gradient(x)*d).sum(-1),scale[...,1],dtype)
    close(2*coeff[...,2],(o.hvp(x,d)*d).sum(-1),2*scale[...,2],dtype)
    close(6*coeff[...,3],(o.third_action(x,d,d)*d).sum(-1),6*scale[...,3],dtype)
    # Independent scalar objective, four autograd derivatives in alpha.
    alpha=torch.tensor(0.,dtype=dtype,requires_grad=True)
    z=o.A@(x[0,0]+alpha*d[0,0])
    c=o.coefficients
    f=-c[0]*torch.dot(o.mu,x[0,0]+alpha*d[0,0])+c[1]*(z**2).mean()-c[2]*(z**3).mean()+c[3]*(z**4).mean()
    for order in range(1,5):
        f=torch.autograd.grad(f,alpha,create_graph=True)[0]
        close(math.factorial(order)*coeff[0,0,order],f,math.factorial(order)*scale[0,0,order],dtype)


def test_zero_direction_cache_batch_and_broadcast(dtype):
    o,x,d=problem(dtype)
    cached=o.cache(x)
    a=mvsk_quartic_coefficients(o,x,d,cache=cached)
    torch.testing.assert_close(a,mvsk_quartic_coefficients(o,x,d),rtol=0,atol=0)
    for i in range(2):
        for j in range(3):
            s=mvsk_quartic_coefficients(o,x[i,j],d[i,j])
            _,scale=expansion(o,x[i,j],d[i,j])
            close(s,a[i,j],scale,dtype)
    z=mvsk_quartic_coefficients(o,x,torch.zeros_like(d))
    torch.testing.assert_close(z[...,0],o.value(x),rtol=0,atol=0)
    assert (z[...,1:]==0).all()
    xsmall=x[:,:1]; c=o.cache(xsmall)
    b=mvsk_quartic_coefficients(o,xsmall,d[0],cache=c)
    assert b.shape==(2,3,5)
    ref,scale=expansion(o,xsmall,d[0]); close(b,ref,scale,dtype)


@pytest.mark.parametrize('scale',[1e-3,1e3])
def test_scale_stress(dtype,scale):
    o,x,d=problem(dtype,scale)
    coeff=mvsk_quartic_coefficients(o,x*.1,d*10)
    ref,magnitude=expansion(o,x*.1,d*10)
    close(coeff,ref,magnitude,dtype)


@pytest.mark.parametrize('weights,index', [([0,1.5,1,0],1),([0,3,1,0],2),([0,1,4,1],3)])
def test_cancellation(dtype,weights,index):
    o=MVSKOracle(MVSKData(torch.tensor([[1.]],dtype=dtype),torch.zeros(1,dtype=dtype),1,1),weights)
    x=torch.ones(1,dtype=dtype); d=x.clone()
    a=mvsk_quartic_coefficients(o,x,d)
    ref,scale=expansion(o,x,d); close(a,ref,scale,dtype)
    assert a[index]==0


@pytest.mark.parametrize('kind',['power','projection','zero_weight'])
def test_overflow(dtype,kind):
    A=torch.tensor([[1.]],dtype=dtype)
    weights=[0,1,1,1] if kind!='zero_weight' else [1,0,0,0]
    if kind=='projection': A*=torch.finfo(dtype).max/2
    o=MVSKOracle(MVSKData(A,torch.zeros(1,dtype=dtype),1,1),weights)
    huge=1e12 if dtype==torch.float32 else 1e100
    with pytest.raises(FloatingPointError):
        mvsk_quartic_coefficients(o,torch.ones(1,dtype=dtype),torch.tensor([huge],dtype=dtype))


@pytest.mark.parametrize('kind',['wrong_owner','new_object','shape_metadata','dtype','asset','nonfinite','batch'])
def test_invalid(kind,dtype):
    from dataclasses import replace
    o,x,d=problem(dtype); cache=o.cache(x)
    if kind=='wrong_owner': cache=replace(cache,owner=object())
    if kind=='new_object': x=x.clone()
    if kind=='shape_metadata': cache=replace(cache,shape=torch.Size([4]))
    if kind=='dtype': d=d.to(torch.float64 if dtype==torch.float32 else torch.float32)
    if kind=='asset': d=d[...,:1]
    if kind=='nonfinite': d=d.clone(); d[0,0,0]=float('nan')
    if kind=='batch': d=d[:,:,:].reshape(6,4)
    with pytest.raises((ValueError,TypeError,RuntimeError)):
        mvsk_quartic_coefficients(o,x,d,cache=cache)


def test_autograd_preserved():
    o,x,d=problem(torch.float64)
    x=x.requires_grad_(); d=d.requires_grad_()
    a=mvsk_quartic_coefficients(o,x,d)
    gx,gd=torch.autograd.grad(a.sum(),(x,d))
    assert torch.isfinite(gx).all() and torch.isfinite(gd).all()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda(dtype):
    o,x,d=problem(dtype)
    gpu=MVSKOracle(MVSKData(o.A.cuda(),o.mu.cuda(),o.T,o.n),o.coefficients.cuda())
    a=mvsk_quartic_coefficients(o,x,d)
    b=mvsk_quartic_coefficients(gpu,x.cuda(),d.cuda()).cpu()
    _,scale=expansion(o,x,d); close(b,a,scale,dtype)
