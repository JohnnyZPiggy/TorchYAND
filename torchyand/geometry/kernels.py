"""Unchecked fixed-shape tensor projectors; public callers validate inputs."""
import torch


def simplex_tangent(v):
    return v-v.mean(dim=-1, keepdim=True)


def face_tangent(v, free_mask):
    masked = torch.where(free_mask, v, 0)
    count = free_mask.sum(dim=-1, keepdim=True)
    mean = masked.sum(dim=-1, keepdim=True)/count.clamp_min(1)
    return torch.where(free_mask & (count > 1), v-mean, 0)


def geometry(gradient, free_mask, grad_tol=None):
    grad_face = face_tangent(gradient, free_mask)
    grad_norm = torch.linalg.vector_norm(grad_face, dim=-1)
    free_count = free_mask.sum(dim=-1)
    # This branch depends only on an optional argument, never tensor values.
    if grad_tol is None:
        masked = torch.where(free_mask, gradient, 0)
        masked_norm = torch.linalg.vector_norm(masked, dim=-1)
        threshold = (128*torch.finfo(gradient.dtype).eps
                     * free_count.clamp_min(1).to(gradient.dtype).sqrt()
                     * masked_norm.clamp_min(1))
    else:
        threshold = grad_tol
    regular = grad_norm > threshold
    denominator = torch.where(regular, grad_norm, 1)
    nu = torch.where(regular.unsqueeze(-1), grad_face/denominator.unsqueeze(-1), 0)
    return grad_face, grad_norm, nu, free_count, regular, threshold


def level_tangent(v, free_mask, nu):
    face = face_tangent(v, free_mask)
    return face-nu*(nu*face).sum(dim=-1, keepdim=True)
