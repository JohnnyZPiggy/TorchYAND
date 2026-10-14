"""Unchecked fixed-shape MVSK line algebra; returns coefficients and finite mask."""
import torch
from ..oracle.kernels import value


def mvsk_quartic_coefficients_unchecked(x, d, z, w, mu, c):
    c1, c2, c3, c4 = c.unbind()
    z2, w2 = z*z, w*w
    z3, z4 = z2*z, z2*z2
    w3, w4 = w2*w, w2*w2
    zw, z2w, z3w = z*w, z2*w, z3*w
    zw2, z2w2, zw3 = z*w2, z2*w2, z*w3
    t11, t12, t13 = 2*c2*zw, -3*c3*z2w, 4*c4*z3w
    t21, t22, t23 = c2*w2, -3*c3*zw2, 6*c4*z2w2
    t31, t32, t4 = -c3*w3, 4*c4*zw3, c4*w4
    # Keep objective evaluation identical to the frozen oracle value kernel.
    a0 = value(x, z, mu, c)
    mean_terms = d*mu
    mean_d = -c1*mean_terms.sum(-1)
    s1, s2, s3 = t11+t12, t21+t22, t31+t32
    v1, v2 = s1+t13, s2+t23
    coeff = torch.stack((a0, mean_d+v1.mean(-1), v2.mean(-1), s3.mean(-1), t4.mean(-1)), -1)
    finite = torch.isfinite(coeff).all(-1)
    # The loop is over a static tuple, never over tensor-dependent data.
    for term in (z2,z3,z4,w2,w3,w4,zw,z2w,z3w,zw2,z2w2,zw3,
                 t11,t12,t13,t21,t22,t23,t31,t32,t4,s1,s2,s3,v1,v2,mean_terms):
        finite = finite & torch.isfinite(term).all(-1)
    return coeff, finite
