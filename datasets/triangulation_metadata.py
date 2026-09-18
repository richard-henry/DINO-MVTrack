"""Keep observed geometry separate from interpolation; never change the init loss."""
from itertools import combinations

import numpy as np


def observed_metadata(tracks, raw_valid, confidence, reprojection_error, num_views,
                      ray_angle, query_points, intrinsics, extrinsics, pseudo_tracks,
                      pseudo_visibility, explicit_raw=None):
    post_query = np.arange(tracks.shape[0])[:, None] > query_points[None, :, 0]
    valid = (raw_valid > 0.5) & post_query
    raw = tracks if explicit_raw is None else explicit_raw
    if raw.shape != tracks.shape or not np.isfinite(raw[valid]).all():
        raise ValueError('Invalid raw triangulation coordinates')
    # Old files only stored filled coordinates. Verify the retained post-query
    # observations against the recorded median reprojection error and support
    # count. Query-overwritten frames cannot be promoted to raw anchors.
    if valid.any():
        hom = np.concatenate([np.nan_to_num(raw), np.ones(raw.shape[:-1] + (1,))], axis=-1)
        projection = intrinsics @ extrinsics[..., :3, :]
        uvh = np.einsum('svij,snj->svni', projection, hom)
        depth = uvh[..., 2]
        uv = uvh[..., :2] / np.where(np.abs(depth[..., None]) > 1e-8, depth[..., None], 1.)
        error = np.linalg.norm(uv - pseudo_tracks.transpose(1, 0, 2, 3), axis=-1)
        visible = (pseudo_visibility.transpose(1, 0, 2) > 0.5) & (depth > 1e-6)
        matched = np.zeros_like(valid)
        for count in range(2, intrinsics.shape[1] + 1):
            for subset in combinations(range(intrinsics.shape[1]), count):
                median = np.median(error[:, subset, :], axis=1)
                matched |= ((num_views == count) & visible[:, subset, :].all(axis=1)
                            & np.isclose(median, reprojection_error, atol=0.05, rtol=0.005))
        if np.any(valid & ~matched):
            raise ValueError('Raw coordinates fail recorded reprojection/support consistency; regenerate init metadata')
    gap = post_query & ~valid
    return {
        'init_raw_tracks_3d': np.where(valid[..., None], raw, np.nan).astype(np.float32),
        'init_raw_confidence': np.where(valid, confidence, 0.).astype(np.float32),
        'init_raw_reproj_error': np.where(valid, reprojection_error, np.nan).astype(np.float32),
        'init_raw_num_views': np.where(valid, num_views, 0.).astype(np.float32),
        'init_raw_ray_angle': np.where(valid, ray_angle, 0.).astype(np.float32),
        'init_completion_mask': gap.astype(np.float32),
        # This is inherited interpolation quality, not observed confidence.
        'init_completion_confidence': np.where(gap, confidence, 0.).astype(np.float32),
    }
