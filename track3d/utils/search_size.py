"""Search-scale time granularity; preserve the original shrink rule for attribution."""
import torch

SEARCH_SIZE_CONTRACT = {
    'version': 'time_granularity_v1',
    'shared': 'size[B,N,3]; max projected span and signed grid delta across time',
    'per_frame': 'size[B,S,N,3]; each frame uses its own projected span and signed grid delta',
    'rule': 'iteration>=2, span>=3, delta<2; shrink by .6 if delta<1 else .8; clamp to original threshold',
    'unchanged': 'initial size, signed delta convention, view max, thresholds, losses, initialization, visibility',
}


def validate_search_size(policy):
    if policy not in ('shared', 'per_frame'):
        raise ValueError(f'Unknown search_size_policy: {policy!r}')


def initialize_search_size(size, frames, policy):
    validate_search_size(policy)
    if size.ndim != 3 or size.shape[-1] != 3:
        raise ValueError('Initial search size must be [B,N,3]')
    if policy == 'shared':
        return size
    return size.unsqueeze(1).expand(-1, frames, -1, -1).clone()


def adapt_search_size(size, view_max_spans, delta, threshold, iteration, adaptive, policy):
    """Reduce time only in shared mode. Signed delta is intentionally unchanged."""
    validate_search_size(policy)
    if delta.ndim != 4 or delta.shape[-1] != 3 or view_max_spans.shape != delta.shape[:-1]:
        raise ValueError('Expected delta[B,S,N,3] and projected spans[B,S,N]')
    if policy == 'shared':
        if size.shape != (delta.shape[0], delta.shape[2], 3):
            raise ValueError('Shared search size must be [B,N,3]')
        span = view_max_spans.max(dim=1)[0]
        update = torch.max(delta, dim=1)[0]
    else:
        if size.shape != delta.shape:
            raise ValueError('Per-frame search size must be [B,S,N,3]')
        span, update = view_max_spans, delta
    is_too_tight = (span < 3.0).unsqueeze(-1)
    adjusted_size = size.clone()
    if adaptive and iteration >= 2:
        shrink_mask = (update < 2.0) & (~is_too_tight)
        shrink_rate = torch.where(update < 1.0, 0.6, 0.8)
        adjusted_size[shrink_mask] = adjusted_size[shrink_mask] * shrink_rate[shrink_mask]
    return torch.where(adjusted_size > threshold, adjusted_size, threshold)
