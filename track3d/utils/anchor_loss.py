"""V02A anchor support ablation; confidence and L1 normalization stay fixed."""
import math
import json
import torch


def validate_anchor(policy, weight=0.0):
    if policy not in ('legacy', 'raw_only', 'weak_geometry'):
        raise ValueError(f'Unknown anchor_policy: {policy!r}')
    if not math.isfinite(float(weight)) or weight < 0:
        raise ValueError('Anchor weight must be finite and nonnegative')


def anchor_metadata(policy):
    if policy != 'weak_geometry': return {}
    from track3d.utils.directional_anchor import CONTRACT
    return dict(anchor_contract_json=json.dumps(CONTRACT, sort_keys=True))


def validate_anchor_checkpoint(checkpoint, policy):
    for key, value in anchor_metadata(policy).items():
        if checkpoint.get(key) != value:
            raise ValueError('Checkpoint directional anchor contract differs')


def triangulation_anchor(prediction, initial, confidence, query_t,
                         policy='legacy', raw_valid=None, raw_tracks=None,
                         raw_confidence=None):
    """Return unweighted mean L1 loss and detached effective support weights.

    Raw-only is strictly a support ablation: observed coordinates/confidence
    must equal the legacy values on retained frames. Missing raw metadata fails
    explicitly. Inactive NaN raw coordinates never enter the loss arithmetic.
    The denominator is each policy's own confidence mass, as in the legacy mean.
    """
    validate_anchor(policy)
    if policy == 'weak_geometry':
        raise ValueError('weak_geometry requires the directional anchor geometry path')
    B, S, N, D = prediction.shape
    if D != 3 or initial.shape != prediction.shape or confidence.shape != (B, S, N) or query_t.shape != (B, N):
        raise ValueError('Anchor coordinate/confidence/query shapes differ')
    target = initial.detach()
    confidence = confidence.detach()
    frames = torch.arange(S, device=prediction.device).view(1, S, 1)
    weights = (frames >= query_t.view(B, 1, N)).float() * confidence.clamp(0.0, 1.0)
    if policy == 'raw_only':
        if raw_valid is None or raw_tracks is None or raw_confidence is None:
            raise ValueError('raw_only requires checked raw mask, coordinates and confidence')
        raw_valid, raw_tracks, raw_confidence = (
            x.detach().to(device=prediction.device) for x in (raw_valid, raw_tracks, raw_confidence))
        if raw_valid.shape != weights.shape or raw_tracks.shape != target.shape or raw_confidence.shape != weights.shape:
            raise ValueError('Raw anchor metadata shapes differ')
        if not torch.isfinite(raw_valid).all():
            raise ValueError('Non-finite raw validity mask')
        observed = (raw_valid > 0.5) & (frames > query_t.view(B, 1, N))
        if not torch.isfinite(raw_tracks[observed]).all() or not torch.isfinite(raw_confidence[observed]).all():
            raise ValueError('Non-finite observed anchor coordinates/confidence')
        if not torch.equal(raw_tracks[observed], target[observed]) or not torch.equal(raw_confidence[observed], confidence[observed]):
            raise ValueError('Support-only ablation requires unchanged observed target/confidence')
        weights = weights * observed
        # Keep the same finite initial target. Equality above proves it is raw
        # at every retained location; no NaN * zero or silent gap fallback.
    distance = torch.norm(prediction - target, p=1, dim=-1)
    loss = (distance * weights).sum() / (weights.sum() + 1e-6)
    return loss, weights
