"""Frozen coordinate hypotheses, explicitly separate from visibility labels."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


CONTRACT = dict(version='retracking_auxiliary_v1', weight=0.1, huber_beta_px=2.0,
    target='unchanged_R02_accepted_coordinates_including_uncertain_occlusion',
    application='final_iteration_only', normalization='fixed_accepted_observation_count',
    projection='float32_positive_depth_unclipped_pixels',
    visibility='original_supervision_forward_and_output_unchanged',
    selection='total_original_plus_auxiliary_loss_with_separate_component_and_coverage',
    sidecar='triangulation_init_root/scene/retracking_candidates.npz', no_additional_gt=True)


def load_candidates(path, sample):
    """Validate clip identity once at dataset loading; never inspect future GT."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        a = {key: data[key] for key in data.files}
    if json.loads(str(a['contract_json'])) != CONTRACT:
        raise ValueError('Auxiliary coordinate contract differs')
    reference = dict(sample_indices=sample['sample_indices'].numpy(),
        query_points=sample['query_points'].numpy(),
        old_uv=sample['cotracker_tracks'].numpy().transpose(1, 0, 2, 3),
        pseudo_visibility=sample['cotracker_vis'].numpy().transpose(1, 0, 2),
        intrinsics=sample['intrinsics'].numpy(), extrinsics=sample['extrinsics'].numpy())
    for key, value in reference.items():
        if not np.array_equal(a[key], value, equal_nan=True):
            raise ValueError('Auxiliary input identity differs: ' + key)
    s, v, n = reference['pseudo_visibility'].shape
    mask = a['accepted']
    uv = a['uv']
    if mask.dtype != np.bool_ or mask.shape != (s, v, n) or uv.shape != (s, v, n, 2):
        raise ValueError('Invalid auxiliary candidate shape or mask')
    qt = reference['query_points'][:, 0].astype(int)
    qv = sample['visibility'].numpy()[qt, np.arange(n)].T
    allowed = ((reference['pseudo_visibility'] <= .5) & (qv[None] > .5)
               & (np.arange(s)[:, None, None] > qt[None, None]))
    h, w = sample['rgbs'].shape[-2:]
    in_frame = (np.isfinite(uv).all(-1) & (uv[..., 0] >= 0) & (uv[..., 0] <= w-1)
                & (uv[..., 1] >= 0) & (uv[..., 1] <= h-1))
    if np.any(mask & ~(allowed & in_frame)):
        raise ValueError('Auxiliary candidate violates original query/pseudo support')
    return dict(retracking_targets=torch.from_numpy(np.where(mask[..., None], uv, 0).astype(np.float32)),
                retracking_mask=torch.from_numpy(mask.copy()),
                retracking_input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def auxiliary_loss(prediction, intrinsics, extrinsics, targets, mask):
    """Radial Huber with bounded image-space influence and fixed denominator.

    Out-of-frame predictions retain gradient. Nonpositive-depth projections do
    not define image coordinates: record their lost coverage for the existing
    stopping/checkpoint guards rather than changing the denominator.
    """
    with torch.amp.autocast(prediction.device.type, enabled=False):
        x = prediction.float()
        k, rt, target = (t.detach().to(x.device).float() for t in (intrinsics, extrinsics, targets))
        active = mask.detach().to(x.device).bool()
        b, s, n, _ = x.shape
        v = k.shape[2]
        if (k.shape != (b,s,v,3,3) or rt.shape != (b,s,v,4,4)
                or target.shape != (b,s,v,n,2) or active.shape != (b,s,v,n)):
            raise ValueError('Auxiliary projection dimensions differ')
        if not torch.isfinite(x).all() or not torch.isfinite(target[active]).all():
            raise ValueError('Nonfinite auxiliary prediction or accepted target')
        hom = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        camera = torch.einsum('bsvij,bsnj->bsvni', rt, hom)[..., :3]
        projected = torch.einsum('bsvij,bsvnj->bsvni', k, camera)
        valid = active & (camera[..., 2] > 0) & (projected[..., 2] > 0) & torch.isfinite(projected).all(-1)
        safe = torch.where(valid[..., None], projected, torch.zeros_like(projected))
        uv = safe[..., :2] / torch.where(valid, projected[..., 2], 1.)[..., None]
        residual = uv - torch.where(valid[..., None], target, torch.zeros_like(target))
        distance = torch.linalg.vector_norm(residual, dim=-1)
        beta = CONTRACT['huber_beta_px']
        penalty = torch.where(distance < beta, .5*distance.square()/beta, distance-.5*beta)
        count = active.sum()
        value = (penalty * valid).sum() / count.clamp_min(1)
        diagnostics = dict(accepted_count=int(count), valid_count=int(valid.sum()),
                           coverage=float(valid.sum()/count.clamp_min(1)) if count else 1.0)
        return value, diagnostics


def add_diagnostics(diagnostics, value, observation):
    """Extend the objective ledger without relabeling auxiliary loss as anchor."""
    result = dict(diagnostics)
    weighted = CONTRACT['weight'] * float(value)
    result.update(schema=3, auxiliary=float(value), weighted_auxiliary=weighted,
                  auxiliary_observations=observation)
    result['reconstructed_total'] += weighted
    result['reconstruction_error'] = result['reconstructed_total']-result['total']
    result['coverage'] = result['coverage'] + [observation['coverage']]
    return result
