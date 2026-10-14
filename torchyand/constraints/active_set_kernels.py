"""Fixed-shape mask-only face updates, with deterministic first-index ties."""
import torch


def update_active_set_unchecked(x, grad, free_mask, request, tau, active_tol, kkt, *, phase):
    # Status: no change, activated, released, preserved, invalid face/request.
    if phase == 'activation':
        invalid_request = (request & (~free_mask | ((x-tau.unsqueeze(-1)).abs() > active_tol.unsqueeze(-1)))).any(-1)
        eligible = request & free_mask
        remove_all = eligible.any(-1) & (eligible.sum(-1) == free_mask.sum(-1))
        scores = torch.where(eligible, x-tau.unsqueeze(-1), float('-inf'))
        keep_index = scores.argmax(-1)
        keep = torch.arange(x.shape[-1],device=x.device) == keep_index.unsqueeze(-1)
        activated = eligible & ~(keep & remove_all.unsqueeze(-1))
        released = torch.zeros_like(free_mask)
        proposed = free_mask & ~activated
        status = torch.where(remove_all, 3, torch.where(activated.any(-1),1,0))
    else:
        invalid_request = (request & free_mask).any(-1)
        eligible = kkt.release_mask & request
        gate = kkt.face_stationary & ~kkt.kkt_stationary
        scores = torch.where(eligible, kkt.simplex_multiplier.unsqueeze(-1)-grad, float('-inf'))
        index = scores.argmax(-1)
        chosen = torch.arange(x.shape[-1],device=x.device) == index.unsqueeze(-1)
        released = chosen & eligible & gate.unsqueeze(-1)
        activated = torch.zeros_like(free_mask)
        proposed = free_mask | released
        status = torch.where(released.any(-1),2,0)
    valid = kkt.valid & ~invalid_request
    activated = activated & valid.unsqueeze(-1)
    released = released & valid.unsqueeze(-1)
    result = torch.where(valid.unsqueeze(-1), proposed, free_mask)
    status = torch.where(~kkt.valid,4,torch.where(invalid_request,5,status))
    changed = (activated | released).any(-1)
    return result, activated, released, changed, valid, status
