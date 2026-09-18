"""Metric names and helpers aligned with common 3D point tracking reports."""

from __future__ import annotations

import numpy as np

THRESHOLDS_BY_SETTING = {
    "kubric-multiview": [0.05, 0.10, 0.20, 0.40, 0.80],
    "pointodyssey-multiview": [0.05, 0.10, 0.20, 0.40, 0.80],
    "dexycb-multiview": [0.01, 0.02, 0.05, 0.10, 0.20],
    "panoptic-multiview": [0.05, 0.10, 0.20, 0.40],
    "tapvid2d": [1, 2, 4, 8, 16],
}

METRIC_GROUPS = [
    "average_jaccard",
    "average_pts_within_thresh",
    "occlusion_accuracy",
    "mte_visible",
    "ate_visible",
    "fde_visible",
    "survival",
]

POINT_GROUPS = ["any", "static", "dynamic", "very_dynamic", "dynamic-static-mean"]

MOTION_THRESHOLDS_BY_SETTING = {
    "kubric-multiview": {
        "static": 0.01,
        "dynamic": 0.10,
        "very_dynamic": 2.0,
        "survival": 0.50,
    },
    "pointodyssey-multiview": {
        "static": 0.01,
        "dynamic": 0.10,
        "very_dynamic": 2.0,
        "survival": 0.50,
    },
    "dexycb-multiview": {
        "static": 0.01,
        "dynamic": 0.10,
        "very_dynamic": 0.50,
        "survival": 0.10,
    },
    "panoptic-multiview": {
        "static": 0.01,
        "dynamic": 0.10,
        "very_dynamic": 0.50,
        "survival": 1.00,
    },
    "tapvid2d": {
        "static": 1.0,
        "dynamic": 1.0,
        "very_dynamic": 50.0,
        "survival": 50.0,
    },
}


def valid_after_query_mask(query_points, num_frames, include_query=False):
    """Return an `(S, N)` mask that ignores frames before each query time."""
    query_t = np.asarray(query_points)[..., 0].astype(np.int64)
    start = query_t if include_query else query_t + 1
    frames = np.arange(num_frames, dtype=np.int64)[:, None]
    return frames >= start[None, :]


def compute_3dpt_metrics(
    pred_tracks,
    pred_visibility,
    gt_tracks,
    gt_visibility,
    query_points=None,
    thresholds=None,
    include_query=False,
    point_mask=None,
    survival_threshold=None,
    mvtracker_strict=False,
    scale_metrics_by_100=False,
):
    """Compute scene-level 3D point tracking metrics.

    Args:
        pred_tracks: `(S, N, 3)` predicted world-space tracks.
        pred_visibility: `(S, N)` or `(S, N, V)` predicted visibility.
        gt_tracks: `(S, N, 3)` ground-truth/reference world-space tracks.
        gt_visibility: `(S, N)` or `(S, N, V)` ground-truth visibility.
        query_points: optional `(N, 4)` query points `(t, x, y, z)`.
        thresholds: distance thresholds in world units.
        include_query: whether to include the query frame in evaluation.
        point_mask: optional `(N,)` boolean mask for motion/visibility groups.
        survival_threshold: optional world-space failure threshold. Defaults to
            the largest entry in `thresholds` for backward compatibility.
        mvtracker_strict: use MVTracker-style per-track aggregation for
            trajectory errors and survival.
        scale_metrics_by_100: multiply ratio metrics and errors by 100 for
            MVTracker table-style output.
    """
    pred_tracks = np.asarray(pred_tracks, dtype=np.float32)
    gt_tracks = np.asarray(gt_tracks, dtype=np.float32)
    pred_vis = _collapse_visibility(pred_visibility)
    gt_vis = _collapse_visibility(gt_visibility)
    gt_vis_for_trajectory = gt_vis.copy()

    if pred_tracks.shape != gt_tracks.shape:
        raise ValueError(f"track shape mismatch: pred {pred_tracks.shape}, gt {gt_tracks.shape}")
    if pred_vis.shape != gt_vis.shape:
        raise ValueError(f"visibility shape mismatch: pred {pred_vis.shape}, gt {gt_vis.shape}")
    if pred_tracks.shape[:2] != gt_vis.shape:
        raise ValueError(f"tracks {pred_tracks.shape[:2]} and visibility {gt_vis.shape} are incompatible")

    if thresholds is None:
        thresholds = THRESHOLDS_BY_SETTING["panoptic-multiview"]
    thresholds = np.asarray(thresholds, dtype=np.float32)
    S, N = gt_vis.shape
    if point_mask is not None:
        point_mask = np.asarray(point_mask, dtype=bool).reshape(-1)
        if point_mask.shape[0] != N:
            raise ValueError(f"point_mask has length {point_mask.shape[0]}, expected {N}")
        if not point_mask.any():
            return _empty_metrics()
        pred_tracks = pred_tracks[:, point_mask]
        gt_tracks = gt_tracks[:, point_mask]
        pred_vis = pred_vis[:, point_mask]
        gt_vis = gt_vis[:, point_mask]
        gt_vis_for_trajectory = gt_vis_for_trajectory[:, point_mask]
        if query_points is not None:
            query_points = np.asarray(query_points)[point_mask]
        S, N = gt_vis.shape

    eval_mask = np.ones((S, N), dtype=bool)
    if query_points is not None:
        eval_mask = valid_after_query_mask(query_points, S, include_query=include_query)

    pred_vis = pred_vis & eval_mask
    gt_vis = gt_vis & eval_mask
    dist = np.linalg.norm(pred_tracks - gt_tracks, axis=-1)

    if mvtracker_strict:
        threshold_metrics = _compute_threshold_metrics_per_track(dist, pred_vis, gt_vis, eval_mask, thresholds)
    else:
        threshold_metrics = _compute_threshold_metrics_global(dist, pred_vis, gt_vis, thresholds)

    visible_dist = dist[np.logical_and(gt_vis, eval_mask)]
    final_mask = _final_visible_mask(gt_vis)

    survival_threshold = float(survival_threshold if survival_threshold is not None else thresholds[-1])
    if mvtracker_strict:
        trajectory_eval_mask = np.ones((S, N), dtype=bool)
        if query_points is not None:
            trajectory_eval_mask = valid_after_query_mask(query_points, S, include_query=True)
        trajectory_metrics = _compute_trajectory_metrics_per_track(
            dist=dist,
            gt_vis=gt_vis_for_trajectory,
            eval_mask=trajectory_eval_mask,
            query_points=query_points,
            survival_threshold=survival_threshold,
        )
    else:
        trajectory_metrics = _compute_trajectory_metrics_global(
            dist=dist,
            pred_vis=pred_vis,
            gt_vis=gt_vis,
            eval_mask=eval_mask,
            final_mask=final_mask,
            survival_threshold=survival_threshold,
        )

    metrics = {
        **threshold_metrics,
        **trajectory_metrics,
        "num_eval_points": int(eval_mask.sum()),
        "num_visible_points": int(gt_vis.sum()),
    }
    if mvtracker_strict:
        metrics["occlusion_accuracy"] = _compute_occlusion_accuracy_per_track(pred_vis, gt_vis, eval_mask)
    else:
        metrics["occlusion_accuracy"] = float(_safe_div((pred_vis == gt_vis)[eval_mask].sum(), eval_mask.sum()))
    if scale_metrics_by_100:
        metrics = _scale_metrics(metrics, scale=100.0)
    return metrics


def compute_grouped_3dpt_metrics(
    pred_tracks,
    pred_visibility,
    gt_tracks,
    gt_visibility,
    query_points=None,
    thresholds=None,
    include_query=False,
    setting="panoptic-multiview",
    mvtracker_strict=False,
    scale_metrics_by_100=False,
):
    """Compute MVTracker-style 3DPT metrics for motion-strength groups."""
    if thresholds is None:
        thresholds = THRESHOLDS_BY_SETTING["panoptic-multiview"]
    thresholds = np.asarray(thresholds, dtype=np.float32)
    motion_cfg = MOTION_THRESHOLDS_BY_SETTING.get(setting, MOTION_THRESHOLDS_BY_SETTING["panoptic-multiview"])
    gt_vis = _collapse_visibility(gt_visibility)
    eval_mask = np.ones(gt_vis.shape, dtype=bool)
    if query_points is not None:
        eval_mask = valid_after_query_mask(query_points, gt_vis.shape[0], include_query=True)
    movement = compute_motion_distance(gt_tracks, gt_vis & eval_mask)
    valid_points = (gt_vis & eval_mask).sum(axis=0) >= 2

    masks = {
        "any": valid_points,
        "static": valid_points & (movement < motion_cfg["static"]),
        "dynamic": valid_points & (movement > motion_cfg["dynamic"]),
        "very_dynamic": valid_points & (movement > motion_cfg["very_dynamic"]),
    }

    grouped = {}
    for group, mask in masks.items():
        if not mask.any():
            continue
        grouped[group] = compute_3dpt_metrics(
            pred_tracks=pred_tracks,
            pred_visibility=pred_visibility,
            gt_tracks=gt_tracks,
            gt_visibility=gt_visibility,
            query_points=query_points,
            thresholds=thresholds,
            include_query=include_query,
            point_mask=mask,
            survival_threshold=motion_cfg["survival"],
            mvtracker_strict=mvtracker_strict,
            scale_metrics_by_100=scale_metrics_by_100,
        )
        grouped[group]["num_tracks"] = int(mask.sum())
        grouped[group]["track_percent"] = float(mask.mean())
        grouped[group]["mean_motion"] = float(movement[mask].mean())

    if "static" in grouped and "dynamic" in grouped:
        grouped["dynamic-static-mean"] = {
            key: float(np.mean([grouped["static"][key], grouped["dynamic"][key]]))
            for key in METRIC_GROUPS
        }
        grouped["dynamic-static-mean"]["num_tracks"] = int(masks["static"].sum() + masks["dynamic"].sum())
        grouped["dynamic-static-mean"]["track_percent"] = float((masks["static"] | masks["dynamic"]).mean())
        grouped["dynamic-static-mean"]["mean_motion"] = float(
            movement[masks["static"] | masks["dynamic"]].mean()
        )

    return grouped


def compute_visibility_grouped_3dpt_metrics(
    pred_tracks,
    pred_visibility,
    gt_tracks,
    gt_visibility,
    query_points=None,
    thresholds=None,
    include_query=False,
    setting="panoptic-multiview",
    mvtracker_strict=False,
    scale_metrics_by_100=False,
):
    """Compute 3DPT metrics for simple query/sequence visibility groups."""
    gt_vis_full = np.asarray(gt_visibility)
    gt_vis = _collapse_visibility(gt_visibility)
    S, N = gt_vis.shape
    if query_points is None:
        query_t = np.zeros(N, dtype=np.int64)
    else:
        query_t = np.asarray(query_points)[:, 0].round().astype(np.int64).clip(0, S - 1)

    if gt_vis_full.ndim == 3:
        query_visible = gt_vis_full[query_t, np.arange(N)].max(axis=-1) > 0.5
        visible_rate = gt_vis_full.max(axis=-1).mean(axis=0)
    else:
        query_visible = gt_vis[query_t, np.arange(N)]
        visible_rate = gt_vis.mean(axis=0)

    masks = {
        "query_visible": query_visible,
        "query_invisible": ~query_visible,
        "mostly_visible": visible_rate >= 0.75,
        "intermittent": (visible_rate > 0.0) & (visible_rate < 0.75),
        "never_visible": visible_rate == 0.0,
    }
    motion_cfg = MOTION_THRESHOLDS_BY_SETTING.get(setting, MOTION_THRESHOLDS_BY_SETTING["panoptic-multiview"])
    grouped = {}
    for group, mask in masks.items():
        if not mask.any():
            continue
        grouped[group] = compute_3dpt_metrics(
            pred_tracks=pred_tracks,
            pred_visibility=pred_visibility,
            gt_tracks=gt_tracks,
            gt_visibility=gt_visibility,
            query_points=query_points,
            thresholds=thresholds,
            include_query=include_query,
            point_mask=mask,
            survival_threshold=motion_cfg["survival"],
            mvtracker_strict=mvtracker_strict,
            scale_metrics_by_100=scale_metrics_by_100,
        )
        grouped[group]["num_tracks"] = int(mask.sum())
        grouped[group]["track_percent"] = float(mask.mean())
    return grouped


def compute_2dpt_metrics(
    pred_tracks,
    pred_visibility,
    gt_tracks,
    gt_visibility,
    query_points=None,
    thresholds=None,
    include_query=False,
    mvtracker_strict=False,
    scale_metrics_by_100=False,
):
    """Compute TAP-Vid-style 2D metrics for one projected view."""
    return compute_3dpt_metrics(
        pred_tracks=pred_tracks,
        pred_visibility=pred_visibility,
        gt_tracks=gt_tracks,
        gt_visibility=gt_visibility,
        query_points=query_points,
        thresholds=THRESHOLDS_BY_SETTING["tapvid2d"] if thresholds is None else thresholds,
        include_query=include_query,
        survival_threshold=MOTION_THRESHOLDS_BY_SETTING["tapvid2d"]["survival"],
        mvtracker_strict=mvtracker_strict,
        scale_metrics_by_100=scale_metrics_by_100,
    )


def compute_motion_distance(gt_tracks, gt_visibility):
    """Return per-track path length over visible consecutive GT samples."""
    gt_tracks = np.asarray(gt_tracks, dtype=np.float32)
    gt_visibility = np.asarray(gt_visibility, dtype=bool)
    movement = np.zeros(gt_tracks.shape[1], dtype=np.float32)
    for point_idx in range(gt_tracks.shape[1]):
        point_track = gt_tracks[gt_visibility[:, point_idx], point_idx]
        if point_track.shape[0] >= 2:
            movement[point_idx] = np.linalg.norm(point_track[1:] - point_track[:-1], axis=-1).sum()
    return movement


def _collapse_visibility(visibility):
    visibility = np.asarray(visibility)
    if visibility.ndim == 3:
        visibility = visibility.max(axis=-1)
    if visibility.ndim != 2:
        raise ValueError(f"expected visibility with 2 or 3 dims, got {visibility.shape}")
    return visibility > 0.5


def _safe_div(num, den):
    return float(num) / float(den + 1e-8)


def _compute_threshold_metrics_global(dist, pred_vis, gt_vis, thresholds):
    metrics = {}
    jaccards = []
    pts_within = []
    for threshold in thresholds:
        within = dist <= threshold
        tp = np.logical_and.reduce((pred_vis, gt_vis, within)).sum()
        fp = np.logical_and(pred_vis, np.logical_or(~gt_vis, ~within)).sum()
        fn = np.logical_and(gt_vis, ~pred_vis).sum()
        jaccard = _safe_div(tp, tp + fp + fn)
        pts = _safe_div(np.logical_and(gt_vis, within).sum(), gt_vis.sum())
        metrics[f"jaccard_{threshold:.2f}"] = float(jaccard)
        metrics[f"pts_within_{threshold:.2f}"] = float(pts)
        jaccards.append(jaccard)
        pts_within.append(pts)
    metrics["average_jaccard"] = float(np.mean(jaccards))
    metrics["average_pts_within_thresh"] = float(np.mean(pts_within))
    return metrics


def _compute_threshold_metrics_per_track(dist, pred_vis, gt_vis, eval_mask, thresholds):
    metrics = {}
    jaccards = []
    pts_within = []
    for threshold in thresholds:
        within = dist < threshold
        per_track_jaccard = []
        per_track_pts = []
        for n in range(gt_vis.shape[1]):
            eval_n = eval_mask[:, n]
            if not eval_n.any():
                continue
            gt_n = gt_vis[:, n]
            pred_n = pred_vis[:, n]
            within_n = within[:, n]
            gt_positive = (gt_n & eval_n).sum()
            if gt_positive == 0:
                continue
            correct = (within_n & gt_n & eval_n).sum()
            true_positive = (within_n & pred_n & gt_n & eval_n).sum()
            false_positive = ((~within_n & pred_n) | (pred_n & ~gt_n))
            false_positive = (false_positive & eval_n).sum()
            per_track_pts.append(_safe_div(correct, gt_positive))
            per_track_jaccard.append(_safe_div(true_positive, gt_positive + false_positive))
        metrics[f"jaccard_{threshold:.2f}"] = float(np.mean(per_track_jaccard)) if per_track_jaccard else 0.0
        metrics[f"pts_within_{threshold:.2f}"] = float(np.mean(per_track_pts)) if per_track_pts else 0.0
        jaccards.append(metrics[f"jaccard_{threshold:.2f}"])
        pts_within.append(metrics[f"pts_within_{threshold:.2f}"])
    metrics["average_jaccard"] = float(np.mean(jaccards))
    metrics["average_pts_within_thresh"] = float(np.mean(pts_within))
    return metrics


def _compute_occlusion_accuracy_per_track(pred_vis, gt_vis, eval_mask):
    values = []
    for n in range(gt_vis.shape[1]):
        eval_n = eval_mask[:, n]
        if not eval_n.any():
            continue
        values.append(_safe_div(((pred_vis[:, n] == gt_vis[:, n]) & eval_n).sum(), eval_n.sum()))
    return float(np.mean(values)) if values else 0.0


def _compute_trajectory_metrics_global(dist, pred_vis, gt_vis, eval_mask, final_mask, survival_threshold):
    visible_dist = dist[np.logical_and(gt_vis, eval_mask)]
    survival_values = []
    for n in range(gt_vis.shape[1]):
        valid_frames = np.where(eval_mask[:, n])[0]
        if len(valid_frames) == 0:
            continue
        visible_frames = gt_vis[valid_frames, n]
        if not visible_frames.any():
            continue
        ok = (dist[valid_frames, n] <= survival_threshold) & pred_vis[valid_frames, n] & visible_frames
        first_bad = np.where(~ok)[0]
        survival_values.append(1.0 if len(first_bad) == 0 else first_bad[0] / len(valid_frames))
    return {
        "mte_visible": float(visible_dist.mean()) if visible_dist.size else 0.0,
        "ate_visible": float(visible_dist.mean()) if visible_dist.size else 0.0,
        "fde_visible": float(dist[final_mask].mean()) if final_mask.any() else 0.0,
        "survival": float(np.mean(survival_values)) if survival_values else 0.0,
    }


def _compute_trajectory_metrics_per_track(dist, gt_vis, eval_mask, query_points, survival_threshold):
    mte_values = []
    ate_values = []
    fde_values = []
    survival_values = []
    S, N = gt_vis.shape
    query_t = np.zeros(N, dtype=np.int64)
    if query_points is not None:
        query_t = np.asarray(query_points)[:, 0].astype(np.int64).clip(0, S - 1)
    for n in range(N):
        visible_eval = gt_vis[:, n] & eval_mask[:, n]
        visible_dist = dist[visible_eval, n]
        if visible_dist.size == 0:
            continue
        mte_values.append(_lower_median(visible_dist))
        ate_values.append(float(np.mean(visible_dist)))
        visible_frames = np.where(visible_eval)[0]
        fde_values.append(float(dist[visible_frames[-1], n]))

        failure = visible_eval & (dist[:, n] > survival_threshold)
        failure_frames = np.where(failure)[0]
        denominator = max(S - int(query_t[n]), 1)
        if len(failure_frames) == 0:
            survival_values.append(1.0)
        else:
            survival_values.append(float((failure_frames[0] - int(query_t[n])) / denominator))
    return {
        "mte_visible": float(np.mean(mte_values)) if mte_values else 0.0,
        "ate_visible": float(np.mean(ate_values)) if ate_values else 0.0,
        "fde_visible": float(np.mean(fde_values)) if fde_values else 0.0,
        "survival": float(np.mean(survival_values)) if survival_values else 0.0,
    }


def _scale_metrics(metrics, scale):
    count_keys = {"num_eval_points", "num_visible_points", "num_tracks"}
    return {
        key: (value if key in count_keys else float(value) * scale)
        for key, value in metrics.items()
    }


def _lower_median(values):
    values = np.sort(np.asarray(values, dtype=np.float32).reshape(-1))
    if values.size == 0:
        return 0.0
    return float(values[(values.size - 1) // 2])


def _empty_metrics():
    metrics = {key: 0.0 for key in METRIC_GROUPS}
    metrics["num_eval_points"] = 0
    metrics["num_visible_points"] = 0
    return metrics


def _final_visible_mask(gt_vis):
    mask = np.zeros_like(gt_vis, dtype=bool)
    for n in range(gt_vis.shape[1]):
        frames = np.where(gt_vis[:, n])[0]
        if len(frames) > 0:
            mask[frames[-1], n] = True
    return mask
