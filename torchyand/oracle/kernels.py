"""Unchecked pure tensor MVSK kernels; public callers validate all inputs.

    A=[T,n], mu=[n], c=[4], x/u/v=[...,n], z=[...,T].
    No tensor scalar reads, validation, detaching, or tensor-dependent branches.
"""
import torch


def projection(x, A):
    return x @ A.T


def value(x, z, mu, c):
    c1, c2, c3, c4 = c.unbind()
    z2 = z*z
    z3 = z2*z
    z4 = z2*z2
    return (-c1*(x*mu).sum(-1) + c2*z2.mean(-1)
            - c3*z3.mean(-1) + c4*z4.mean(-1))


def gradient(z, A, mu, c):
    c1, c2, c3, c4 = c.unbind()
    weight = z*(2*c2 + z*(-3*c3 + 4*c4*z))
    return -c1*mu + (weight @ A)/A.shape[0]


def value_grad(x, z, A, mu, c):
    return value(x, z, mu, c), gradient(z, A, mu, c)


def hvp(z, v, A, c):
    _, c2, c3, c4 = c.unbind()
    av = v @ A.T
    weight = 2*c2 + z*(-6*c3 + 12*c4*z)
    return ((weight*av) @ A)/A.shape[0]


def third_action(z, u, v, A, c):
    _, _, c3, c4 = c.unbind()
    au, av = u @ A.T, v @ A.T
    weight = -6*c3 + 24*c4*z
    return ((weight*au*av) @ A)/A.shape[0]
