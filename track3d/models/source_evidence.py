"""Independent frozen DINO templates as auxiliary evidence on shared 3D candidates.

Keep the legacy averaged template and view weights. Reduce each source over
TARGET views first, then summarize usable QUERY sources; no GT or hard source
selection drives the coordinate readout. Resize and sampling match the existing
correlation geometry, including zero padding and align_corners=True.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from track3d.utils.deterministic_sampling import sample_2d

CONTRACT = dict(version='source_evidence_residual_v1',
    base='unchanged UR03 mean of legacy DINO and fnet volumes',
    evidence=['mean_source_score', 'max_source_score', 'source_score_range', 'usable_source_fraction'],
    aggregation='existing DINO target-view weights, then masked query-source statistics',
    source_mask='query visibility >0.5, query projection valid, finite nonzero descriptor',
    sampling='chunked native map resize then existing bilinear sampling; coordinate gradients retained',
    adapter='4-to-32 bias-free 1x1x1 convolution, zero initialized, added before first GroupNorm',
    extra_parameters=128, frozen_dino=True, calibrated_confidence=False)


def source_templates(features, visibility, projection_valid):
    if features.ndim != 4 or features.shape[:-1] != visibility.shape or visibility.shape != projection_valid.shape:
        raise ValueError('Expected B,V,N,C descriptors and B,V,N masks')
    finite = features.isfinite().all(-1)
    clean = torch.where(finite[..., None], features, torch.zeros_like(features))
    valid = (visibility > .5) & projection_valid.bool() & finite & (clean.norm(dim=-1) > 1e-8)
    return F.normalize(clean, dim=-1) * valid[..., None], valid


def summarize_sources(scores, usable):
    """scores B,S,N,Q,G; usable B,Q,N. Permutation-invariant source statistics."""
    if scores.ndim != 5 or usable.shape != (scores.shape[0], scores.shape[3], scores.shape[2]):
        raise ValueError('Mismatched source scores/masks')
    mask = usable.permute(0, 2, 1)[:, None, :, :, None].bool()
    count = mask.sum(3)
    mean = torch.where(mask, scores, torch.zeros_like(scores)).sum(3) / count.clamp_min(1)
    maximum = scores.masked_fill(~mask, -torch.inf).amax(3)
    minimum = scores.masked_fill(~mask, torch.inf).amin(3)
    maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
    minimum = torch.where(count > 0, minimum, torch.zeros_like(minimum))
    fraction = (count.to(scores.dtype) / scores.shape[3]).expand_as(mean)
    return torch.stack((mean, maximum, maximum - minimum, fraction), dim=3)


def _resize_sample(maps, grid, output_size, sampling_policy):
    dense = F.interpolate(maps[:, None].float(), size=output_size, mode='bilinear', align_corners=True)
    h, w = output_size
    norm = torch.stack((2 * grid[..., 0] / (w - 1) - 1,
                        2 * grid[..., 1] / (h - 1) - 1), dim=-1)
    return sample_2d(dense, norm[:, :, None], sampling_policy=sampling_policy)[:, 0, :, 0]


def sample_source_evidence(bank, usable, grid, weights, output_size, sampling_policy, chunk=128):
    """bank Q,B,S,V,N,h,w (frozen); grid B,S,V,N,G,2; weights B,S,N,V.

    Only one chunk of full-resolution maps is materialized at once. Activation
    checkpointing recomputes it for coordinate gradients, rather than retaining
    four full extra high-resolution map banks across the iterative graph.
    """
    if chunk < 1 or bank.requires_grad:
        raise ValueError('Positive chunk and frozen DINO bank required')
    q, b, s, v, n, h, w = bank.shape
    g = grid.shape[-2]
    flat_grid = grid.reshape(b*s*v*n, g, 2)
    source_scores = []
    for source in bank:
        flat_maps = source.reshape(b*s*v*n, h, w)
        sampled = []
        for start in range(0, flat_maps.shape[0], chunk):
            args = (flat_maps[start:start+chunk], flat_grid[start:start+chunk])
            def operation(m, xy):
                return _resize_sample(m, xy, output_size, sampling_policy)
            if torch.is_grad_enabled() and args[1].requires_grad:
                values = checkpoint(operation, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                values = operation(*args)
            sampled.append(values)
        values = torch.cat(sampled).reshape(b,s,v,n,g).permute(0,1,3,2,4)
        values = torch.nan_to_num(values, nan=0., posinf=1., neginf=-1.).clamp(-1.,1.)
        source_scores.append((values * weights[...,None]).sum(3))
    return summarize_sources(torch.stack(source_scores, dim=3), usable)
