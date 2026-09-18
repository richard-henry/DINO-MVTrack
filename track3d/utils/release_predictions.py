"""Explicit delivery arrays derived from predictions and camera parameters only."""
from pathlib import Path
import numpy as np
from track3d.cli.eval import _project_world_to_views


def export_predictions(prediction, camera_file, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    with np.load(prediction, allow_pickle=False) as z:
        xyz = z['trajs_3d'][0]
        visibility = z['visibility']
        query = z['query_points'][0]
        ids = z['sample_indices'].reshape(-1)
        step = z['step'].copy()
        visibility_source = z['visibility_source'].copy()
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or visibility.shape != xyz.shape[:2]:
        raise ValueError('Expected [1,S,N,3] tracks and [S,N] any-view visibility')
    s, n, _ = xyz.shape
    if query.shape != (n, 4) or ids.shape != (n,) or not np.isfinite(xyz).all():
        raise ValueError('Invalid prediction/query identity')
    with np.load(camera_file, allow_pickle=False) as z:
        # Deliberately never read ground-truth tracks or visibility.
        k, rt = z['cam_k'][:s], z['cam_rt'][:s]
        views = z['selected_views'].copy() if 'selected_views' in z else np.arange(k.shape[1])
    if k.shape[0] != s or rt.shape[:2] != k.shape[:2] or len(views) != k.shape[1]:
        raise ValueError('Camera time/view dimensions differ')
    uv, depth = _project_world_to_views(xyz, k, rt)
    uv, depth = uv.transpose(1, 0, 2, 3), depth.transpose(1, 0, 2)
    valid = (depth > 1e-6) & np.isfinite(uv).all(-1)
    uv = np.where(valid[..., None], uv, np.nan)
    active = np.arange(s)[:, None] >= query[None, :, 0]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, tracks_3d=xyz, tracks_2d=uv,
        visibility_any_view=visibility, projection_valid=valid, query_active=active,
        query_points=query, sample_indices=ids, selected_views=views, step=step,
        visibility_source=visibility_source, camera_depth=depth,
        contract=np.array('phase3_delivery_v1'),
        projection_valid_semantics=np.array('finite_positive_depth_only; not image bounds or per-view occlusion'))
    return dict(tracks_3d=list(xyz.shape), tracks_2d=list(uv.shape),
                visibility_any_view=list(visibility.shape), selected_views=views.tolist(),
                positive_depth_finite_projections=int(valid.sum()), total_projections=int(valid.size),
                unique_point_ids=int(len(np.unique(ids))), gt_used=False)
