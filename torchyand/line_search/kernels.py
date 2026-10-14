"""Fixed-shape feasible line-search composition of frozen tensor kernels."""
import torch
from ..oracle.kernels import value
from ..line_models.kernels import mvsk_quartic_coefficients_unchecked
from ..polynomial.quartic_line_search import _minimize_quartic_interval_unchecked
from ..constraints.kernels import face_validity


def exact_mvsk_line_search_unchecked(x, z, cap, A, mu, c, tau, *,
                                    interval_tol=None, value_tol=None,
                                    acceptance_tol=None, run_quartic=True):
    eps = torch.finfo(x.dtype).eps
    q, upper = cap.direction, cap.alpha_max
    zero_direction = cap.zero_direction & torch.isposinf(upper)
    zero_cap = upper == 0
    normal = cap.success & ~zero_direction & ~zero_cap & torch.isfinite(upper) & (upper > 0)
    status = torch.where(~cap.success, 4, torch.where(~zero_direction & ~torch.isfinite(upper), 5,
                         torch.where(zero_direction, 1, torch.where(zero_cap, 2, 0))))
    # Current objective is at the original x. No trial may reuse this z.
    current = value(x, z, mu, c)
    safe_x = torch.where(normal.unsqueeze(-1), x, 0)
    safe_z = torch.where(normal.unsqueeze(-1), z, 0)
    safe_q = torch.where(normal.unsqueeze(-1), q, 0)
    w = safe_q @ A.T
    coeff, assembled = mvsk_quartic_coefficients_unchecked(safe_x, safe_q, safe_z, w, mu, c)
    assembled &= torch.isfinite(z).all(-1) & torch.isfinite(w).all(-1) & torch.isfinite(current)
    status = torch.where(normal & ~assembled, 6, status)
    eligible = normal & assembled
    dummy = torch.zeros_like(coeff)
    dummy = torch.cat((torch.where(torch.isfinite(current),current,0).unsqueeze(-1),dummy[...,1:]),-1)
    safe_coeff = torch.where(eligible.unsqueeze(-1), coeff, dummy)
    safe_upper = torch.where(eligible, upper, 0)
    # run_quartic is a static public scheduling choice, not a tensor branch.
    quartic = None
    if run_quartic:
        quartic = _minimize_quartic_interval_unchecked(safe_coeff, torch.zeros_like(upper), safe_upper,
                                                     interval_tol=interval_tol, value_tol=value_tol)
        boundary_tol = 64*eps*safe_upper.abs().clamp_min(1) if interval_tol is None else interval_tol
        interval_ok = torch.isfinite(quartic.x) & (quartic.x >= -boundary_tol) & (quartic.x <= safe_upper+boundary_tol)
        quartic_ok = quartic.success & interval_ok
        status = torch.where(eligible & ~quartic_ok, 7, status)
        eligible &= quartic_ok
        alpha = torch.where(eligible, quartic.x, 0)
    else:
        alpha = torch.zeros_like(upper)
    # Full Horner evaluation and cancellation-aware contribution scales.
    p = safe_coeff[...,4]
    for i in (3,2,1,0):
        p = p*alpha+safe_coeff[...,i]
    scale = safe_coeff[...,0].abs().clamp_min(1)
    power = torch.ones_like(alpha)
    contribution_finite = torch.ones_like(eligible)
    for i in (1,2,3,4):
        power = power*alpha
        term = safe_coeff[...,i]*power
        contribution_finite &= torch.isfinite(term)
        scale = torch.maximum(scale, term.abs())
    predicted = torch.where(normal, p, current)
    poly_ok = contribution_finite & torch.isfinite(p) & ((p-current) <= 128*eps*scale)
    status = torch.where(eligible & ~poly_ok, 9, status)
    eligible &= poly_ok
    proposed = x+alpha.unsqueeze(-1)*q
    finite_trial = torch.isfinite(proposed).all(-1)
    status = torch.where(eligible & ~finite_trial, 9, status)
    eligible &= finite_trial
    finite_proposed = torch.where(finite_trial.unsqueeze(-1), proposed, x)
    feasible, _ = face_validity(finite_proposed, tau, cap._free_mask)
    slack = finite_proposed-tau.unsqueeze(-1)
    active_ok = torch.where(cap._free_mask, True, slack.abs() <= 64*eps).all(-1)
    status = torch.where(eligible & ~(feasible & active_ok), 8, status)
    eligible &= feasible & active_ok
    special = cap.success & (zero_direction | zero_cap)
    verify = eligible | special
    evaluation_x = torch.where(eligible.unsqueeze(-1), proposed, x)
    trial_z = evaluation_x @ A.T
    trial = value(evaluation_x, trial_z, mu, c)
    acceptance = 128*eps*torch.maximum(current.abs(),trial.abs()).clamp_min(1) if acceptance_tol is None else acceptance_tol
    verification_scale = torch.maximum(scale,trial.abs()).clamp_min(1)
    verified = (torch.isfinite(current) & torch.isfinite(trial) & torch.isfinite(trial_z).all(-1)
                & torch.isfinite(predicted) & ((trial-current) <= acceptance)
                & ((trial-predicted).abs() <= 256*eps*verification_scale))
    status = torch.where(verify & ~verified, 9, status)
    success = verify & verified
    zero_tol = 64*eps*torch.ones_like(alpha) if interval_tol is None else interval_tol
    status = torch.where(success & normal & (alpha.abs() <= zero_tol), 3, status)
    final_alpha = torch.where(success,alpha,0)
    final_x = torch.where(success.unsqueeze(-1),proposed,x)
    bound_tol = 64*eps*torch.maximum(alpha.abs(),torch.where(torch.isfinite(upper),upper.abs(),0)).clamp_min(1) if interval_tol is None else interval_tol
    at_upper = torch.isfinite(upper) & ((alpha-upper).abs() <= bound_tol)
    hit = cap.hit_mask & (success & (final_alpha > 0) & at_upper).unsqueeze(-1)
    return (final_alpha, final_x, current, torch.where(success,trial,current), predicted,
            q, upper, hit, hit.any(-1), success, status, quartic)
