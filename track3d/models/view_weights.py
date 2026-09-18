"""Isolated choice of the fnet view-weight source; masking stays at the caller."""
import torch.nn.functional as F

POLICIES = ('legacy', 'independent')


def validate_view_weight_policy(policy):
    if policy not in POLICIES:
        raise ValueError('Unknown view weight policy: '+str(policy))


def fnet_view_weights(scores, guided_weights, policy, use_dino, tau=0.1):
    """Return raw weights[B,S,N,V] for query-masked fnet scores[...,V,G3].

    Independent weights use this forward's fnet peak over the SAME candidate
    grid. Keep gradients through that peak/softmax during scene TTO. No extra
    confidence calibration or masking is introduced. With DINO disabled the
    caller's legacy weights already come from fnet, so return them exactly.
    """
    validate_view_weight_policy(policy)
    if policy == 'legacy' or not use_dino:
        return guided_weights
    peak, _ = scores.max(dim=-1)
    return F.softmax(peak / tau, dim=-1)
