"""Center-local residual with a continuous route to the existing distant peak.

The routing weight is a learned-score parametrization, not calibrated confidence.
No absolute scene units, extra model parameters, or GT-derived gates are used.
"""
import torch

POLICIES=('legacy','center_evidence')
CONTRACT=dict(version='center_evidence_v1',center_window=3,
    route='1-exp(-(peak_score-center_score)/existing_iteration_temperature)',
    center='current_position_plus_center_softmax_minus_uniform_residual',
    distant='unchanged_legacy_peak_27_neighborhood_readout',
    flat='exact_identity_with_center_neighborhood_score_gradients',
    additional_parameters=0,calibrated_confidence=False)


def validate_coordinate_update(policy):
    if policy not in POLICIES:raise ValueError('Unknown coordinate update policy: '+str(policy))


def center_evidence_update(confidence,world_volume,current,peak_proposal,temperature):
    """Return coordinates and routing weights, preserving the original dtype.

    An odd symmetric lattice is required. The center softmax residual subtracts
    its uniform counterpart so a constant score volume yields exact identity.
    For a clearly higher distant peak the legacy proposal remains reachable.
    Competing peaks still inherit the existing hard-argmax tie convention.
    """
    if current.shape!=peak_proposal.shape or current.shape[-1]!=3 or temperature<=0:
        raise ValueError('Invalid center update inputs')
    g=confidence.shape[-1]
    if g<3 or g%2!=1 or confidence.shape[-3:]!=(g,g,g):
        raise ValueError('Center update requires an odd cubic grid >=3')
    count=current.numel()//3
    if confidence.numel()!=count*g**3 or world_volume.numel()!=count*g**3*3:
        raise ValueError('Coordinate/confidence shapes differ')
    scores=confidence.reshape(count,g,g,g)
    world=world_volume.reshape(count,g,g,g,3)
    center=current.reshape(count,3);c=g//2
    local=scores[:,c-1:c+2,c-1:c+2,c-1:c+2].reshape(count,27)
    offset=world[:,c-1:c+2,c-1:c+2,c-1:c+2].reshape(count,27,3)-center[:,None]
    probability=torch.softmax(local/temperature,dim=-1)
    uniform=torch.softmax(torch.zeros_like(local),dim=-1)
    local_residual=((probability-uniform)[...,None]*offset).sum(1)
    margin=(scores.reshape(count,-1).amax(-1)-scores[:,c,c,c]).clamp_min(0)
    route=-torch.expm1(-margin/temperature)
    displacement=peak_proposal.reshape(count,3)-center
    updated=center+(1-route[:,None])*local_residual+route[:,None]*displacement
    return updated.reshape_as(current),route.reshape(current.shape[:-1])
