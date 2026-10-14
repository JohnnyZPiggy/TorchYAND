"""Independent scalar objective and autograd derivatives, tests only."""
import torch


def scalar_objective(x, A, mu, c):
    z = A @ x
    return (-c[0]*torch.dot(mu, x) + c[1]*torch.mean(z**2)
            - c[2]*torch.mean(z**3) + c[3]*torch.mean(z**4))


def derivatives(x, u, v, A, mu, c):
    f = lambda point: scalar_objective(point, A, mu, c)
    grad_fn = torch.func.grad(f)
    grad = grad_fn(x)
    hvp_fn = lambda point: torch.func.jvp(grad_fn, (point,), (v,))[1]
    hvp = hvp_fn(x)
    third = torch.func.jvp(hvp_fn, (x,), (u,))[1]
    return f(x), grad, hvp, third
