"""Opt-in score skip connection; neither score is a calibrated probability."""
POLICIES = ('legacy', 'mean_residual', 'range_bounded_residual', 'source_evidence_residual')
CONTRACT = dict(version='mean_score_residual_v1',
    base='arithmetic mean of the two existing view-aggregated correlation volumes',
    residual='original three-layer CNN with zero-initialized output weight and bias',
    output='base plus learned residual, followed by unchanged existing clamp/readout',
    extra_parameters=0, required_branches=['dino', 'fnet'],
    calibrated_confidence=False)

BOUNDED_CONTRACT = dict(version='range_bounded_score_residual_v1',
    base='arithmetic mean of the two existing view-aggregated correlation volumes',
    radius='0.25 times detached per-volume max(base)-min(base)',
    residual='radius * tanh(original CNN residual / radius); zero when radius is zero',
    output='base plus bounded residual, followed by unchanged existing clamp/readout',
    extra_parameters=0, range_fraction=0.25, required_branches=['dino', 'fnet'],
    calibrated_confidence=False)


def validate_score_fusion(policy):
    if policy not in POLICIES:
        raise ValueError('Unknown score fusion policy: '+str(policy))


def fuse_scores(inputs, residual, policy):
    validate_score_fusion(policy)
    if policy == 'legacy':
        return residual
    if inputs.ndim != 5 or inputs.shape[1] != 2 or residual.shape != inputs[:, :1].shape:
        raise ValueError('Score residual expects two 3D volumes and one residual')
    base = inputs.mean(dim=1, keepdim=True)
    if policy in ('mean_residual', 'source_evidence_residual'):
        return base + residual
    # Stop the encoder from increasing its permitted correction through this
    # scale path. The ordinary base and CNN input gradient paths remain open.
    radius = 0.25 * (base.amax(dim=(-3,-2,-1), keepdim=True)
                     - base.amin(dim=(-3,-2,-1), keepdim=True)).detach()
    # No absolute epsilon: exact zero evidence stays flat; small nonzero scales
    # retain unit residual derivative at zero. Inputs here are FP32 in Pips.
    safe_radius = radius.where(radius > 0, radius.new_ones(()))
    return base + radius * (residual / safe_radius).tanh()
