"""View-specific semantic query descriptors with an explicit cross-view fallback."""
import torch
import torch.nn.functional as F

POLICIES = ('legacy', 'per_view')


def validate_query_policy(policy):
    if policy not in POLICIES:
        raise ValueError('Unknown DINO query policy: '+str(policy))


def per_view_templates(features, query_visibility, projection_valid):
    """Return templates[B,V,N,C], same-view mask[B,V,N], fallback valid[B,N].

    Use a normalized same-view descriptor when query-visible (>0.5), in front
    of the camera/in frame, finite and nonzero. Otherwise use the normalized
    visibility-weighted mean of usable query views. Pool BEFORE normalizing,
    as in legacy. No usable fallback returns zero, not fabricated evidence.
    Query visibility follows the existing benchmark input contract; this does
    not infer future visibility or update templates during tracking.
    """
    if features.ndim != 4 or query_visibility.shape != features.shape[:-1] or projection_valid.shape != features.shape[:-1]:
        raise ValueError('Expected descriptors[B,V,N,C] and query masks[B,V,N]')
    finite = torch.isfinite(features).all(-1)
    clean = torch.where(finite[...,None], features, torch.zeros_like(features))
    usable = (query_visibility > .5) & projection_valid.bool() & finite & (clean.norm(dim=-1) > 1e-8)
    weights = torch.where(usable, query_visibility, torch.zeros_like(query_visibility))
    pooled = (clean * weights[...,None]).sum(1) / (weights.sum(1)[...,None] + 1e-6)
    fallback_valid = (weights.sum(1) > 0) & (pooled.norm(dim=-1) > 1e-8)
    pooled = F.normalize(pooled, p=2, dim=-1)
    templates = torch.where(usable[...,None], F.normalize(clean,p=2,dim=-1), pooled[:,None])
    return templates, usable, fallback_valid
