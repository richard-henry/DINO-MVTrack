"""Evaluate Track3D-TTO predictions against exported scene `track.npz` files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from track3d.evaluation.metrics_3dpt import (
    METRIC_GROUPS,
    MOTION_THRESHOLDS_BY_SETTING,
    THRESHOLDS_BY_SETTING,
    compute_2dpt_metrics,
    compute_3dpt_metrics,
    compute_grouped_3dpt_metrics,
    compute_visibility_grouped_3dpt_metrics,
)


PRED_TRACK_KEYS = (
    "trajs_3d",
    "tracks_3d",
    "pred_tracks_3d",
    "pred_trajectories_3d",
    "predictions_worldspace",
    "trajs_e_3d",
    "traj_e",
    "pred_tracks",
    "pred_trajectories",
    "trajectories",
)
PRED_VIS_KEYS = (
    "visibility",
    "pred_visibility",
    "pred_visibilities_any_view",
    "vis",
    "vis_e",
    "vis_prob",
    "vis_prob_dino",
)
PRED_OCC_KEYS = ("pred_occluded", "occluded", "occlusions")
_MISSING = object()


def main(
    prediction_root: str = "outputs",
    dataset_root: str = "",
    output_dir: str = "outputs/eval",
    setting: str = "panoptic-multiview",
    include_query: bool = False,
    scene_id: str = "",
    scene_ids: str = "",
    prediction_name: str = "final_tracks.npz",
    eval_2d_projection: bool = True,
    mvtracker_strict: bool = False,
    scale_metrics_by_100: bool = False,
) -> None:
    if not dataset_root:
        raise ValueError("--dataset_root is required for evaluation")

    prediction_root_path = Path(prediction_root)
    dataset_root_path = Path(dataset_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    thresholds = THRESHOLDS_BY_SETTING.get(setting)
    if thresholds is None:
        known = ", ".join(sorted(THRESHOLDS_BY_SETTING))
        raise ValueError(f"Unknown setting '{setting}'. Known settings: {known}")

    if scene_ids:
        requested = _parse_scene_ids(scene_ids)
        scene_dirs = [dataset_root_path / item for item in requested]
    elif scene_id:
        scene_dirs = [dataset_root_path / str(scene_id)]
    else:
        scene_dirs = sorted(p for p in dataset_root_path.iterdir() if p.is_dir())

    rows = []
    motion_rows = []
    visibility_rows = []
    projection_rows = []
    for scene_dir in scene_dirs:
        if not scene_dir.exists():
            print(f"[skip] scene not found: {scene_dir}")
            continue
        pred_file = _find_prediction_file(
            prediction_root_path,
            scene_dir.name,
            allow_direct=bool(scene_id),
            prediction_name=prediction_name,
        )
        if pred_file is None:
            print(f"[skip] no prediction found for {scene_dir.name}")
            continue

        result = evaluate_scene(
            scene_dir,
            pred_file,
            thresholds,
            include_query,
            prediction_root_path,
            setting=setting,
            eval_2d_projection=eval_2d_projection,
            mvtracker_strict=mvtracker_strict,
            scale_metrics_by_100=scale_metrics_by_100,
        )
        row = result["scene"]
        row["scene_id"] = scene_dir.name
        row["prediction"] = str(pred_file)
        rows.append(row)
        for group, metrics in result["motion"].items():
            motion_rows.append(_named_row(scene_dir.name, pred_file, "point_group", group, metrics))
        for group, metrics in result["visibility"].items():
            visibility_rows.append(_named_row(scene_dir.name, pred_file, "visibility_group", group, metrics))
        for view_name, metrics in result["projection_2d"].items():
            projection_rows.append(_named_row(scene_dir.name, pred_file, "view", view_name, metrics))
        print(f"[ok] {scene_dir.name}: AJ={row['average_jaccard']:.4f}, MTE={row['mte_visible']:.4f}")

    if not rows:
        raise RuntimeError(f"No scenes evaluated from {dataset_root_path}")

    csv_path = output_path / "per_sequence_metrics.csv"
    json_path = output_path / "average_metrics.json"
    md_path = output_path / "comparison_table.md"
    motion_csv_path = output_path / "per_motion_metrics_3dpt.csv"
    motion_json_path = output_path / "average_motion_metrics_3dpt.json"
    visibility_csv_path = output_path / "per_visibility_metrics_3dpt.csv"
    visibility_json_path = output_path / "average_visibility_metrics_3dpt.json"
    projection_csv_path = output_path / "per_view_projection_metrics_2dpt.csv"
    projection_json_path = output_path / "average_projection_metrics_2dpt.json"

    _write_csv(csv_path, rows)
    averages = _average_rows(rows)
    with open(json_path, "w") as f:
        json.dump(averages, f, indent=2, sort_keys=True)
    _write_markdown(md_path, rows, averages)
    _write_group_csv(motion_csv_path, motion_rows, "point_group")
    _write_group_csv(visibility_csv_path, visibility_rows, "visibility_group")
    _write_json(motion_json_path, _average_group_rows(motion_rows, "point_group"))
    _write_json(visibility_json_path, _average_group_rows(visibility_rows, "visibility_group"))
    if projection_rows:
        _write_group_csv(projection_csv_path, projection_rows, "view")
        _write_json(projection_json_path, _average_group_rows(projection_rows, "view"))

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {motion_csv_path}")
    print(f"Wrote {visibility_csv_path}")
    if projection_rows:
        print(f"Wrote {projection_csv_path}")


def evaluate_scene(
    scene_dir,
    pred_file,
    thresholds,
    include_query,
    prediction_root=None,
    setting="panoptic-multiview",
    eval_2d_projection=True,
    mvtracker_strict=False,
    scale_metrics_by_100=False,
):
    gt = np.load(scene_dir / "track.npz", allow_pickle=True)
    pred = np.load(pred_file, allow_pickle=True)

    gt_tracks = gt["trajs_3d"]
    gt_vis = gt["visibility"]
    query_points = gt["query_points"] if "query_points" in gt else _default_query_points(gt_tracks)
    track_upscaling_factor = float(gt["track_upscaling_factor"]) if "track_upscaling_factor" in gt else 1.0

    pred_tracks = _squeeze_batch(_read_first_key(pred, PRED_TRACK_KEYS))
    pred_vis = _read_first_key(pred, PRED_VIS_KEYS, default=None)
    if pred_vis is None:
        pred_occ = _read_first_key(pred, PRED_OCC_KEYS, default=None)
        if pred_occ is None:
            pred_vis = np.ones(gt_vis.shape[:2], dtype=np.float32)
        else:
            pred_vis = 1.0 - _squeeze_batch(pred_occ).astype(np.float32)
    else:
        pred_vis = _squeeze_batch(pred_vis)
    pred_vis = _normalize_prediction_visibility(pred_vis, gt_vis)

    sample_indices = _read_sample_indices(pred, prediction_root, scene_dir.name)
    if sample_indices is not None:
        gt_tracks = gt_tracks[:, sample_indices]
        gt_vis = gt_vis[:, sample_indices]
        query_points = query_points[sample_indices]

    pred_tracks, pred_vis, gt_tracks, gt_vis, query_points = _align_shapes(
        pred_tracks,
        pred_vis,
        gt_tracks,
        gt_vis,
        query_points,
    )

    metric_gt_tracks, metric_pred_tracks, metric_query_points = _scale_tracks_for_3d_metrics(
        gt_tracks,
        pred_tracks,
        query_points,
        track_upscaling_factor,
    )

    scene_metrics = compute_3dpt_metrics(
        pred_tracks=metric_pred_tracks,
        pred_visibility=pred_vis,
        gt_tracks=metric_gt_tracks,
        gt_visibility=gt_vis,
        query_points=metric_query_points,
        thresholds=thresholds,
        include_query=include_query,
        survival_threshold=_survival_threshold_for_setting(setting),
        mvtracker_strict=mvtracker_strict,
        scale_metrics_by_100=scale_metrics_by_100,
    )
    motion_metrics = compute_grouped_3dpt_metrics(
        pred_tracks=metric_pred_tracks,
        pred_visibility=pred_vis,
        gt_tracks=metric_gt_tracks,
        gt_visibility=gt_vis,
        query_points=metric_query_points,
        thresholds=thresholds,
        include_query=include_query,
        setting=setting,
        mvtracker_strict=mvtracker_strict,
        scale_metrics_by_100=scale_metrics_by_100,
    )
    visibility_metrics = compute_visibility_grouped_3dpt_metrics(
        pred_tracks=metric_pred_tracks,
        pred_visibility=pred_vis,
        gt_tracks=metric_gt_tracks,
        gt_visibility=gt_vis,
        query_points=metric_query_points,
        thresholds=thresholds,
        include_query=include_query,
        setting=setting,
        mvtracker_strict=mvtracker_strict,
        scale_metrics_by_100=scale_metrics_by_100,
    )
    projection_metrics = {}
    if eval_2d_projection and _has_camera(gt):
        projection_metrics = _evaluate_projection_2d(
            pred_tracks=pred_tracks,
            pred_visibility=pred_vis,
            gt_tracks=gt_tracks,
            gt_visibility=gt_vis,
            query_points=query_points,
            intrinsics=gt["cam_k"],
            extrinsics=gt["cam_rt"],
            include_query=include_query,
            mvtracker_strict=mvtracker_strict,
            scale_metrics_by_100=scale_metrics_by_100,
        )

    return {
        "scene": scene_metrics,
        "motion": motion_metrics,
        "visibility": visibility_metrics,
        "projection_2d": projection_metrics,
    }


def _survival_threshold_for_setting(setting):
    cfg = MOTION_THRESHOLDS_BY_SETTING.get(setting)
    return None if cfg is None else float(cfg["survival"])


def _scale_tracks_for_3d_metrics(gt_tracks, pred_tracks, query_points, factor):
    if factor == 1.0:
        return gt_tracks, pred_tracks, query_points
    metric_query_points = query_points.copy()
    metric_query_points[:, 1:] = metric_query_points[:, 1:] * factor
    return gt_tracks * factor, pred_tracks * factor, metric_query_points


def _find_prediction_file(prediction_root, scene_id, allow_direct=False, prediction_name="final_tracks.npz"):
    prediction_name = prediction_name or "final_tracks.npz"
    candidates = [
        prediction_root / scene_id / "predictions" / prediction_name,
        prediction_root / scene_id / prediction_name,
        prediction_root / scene_id / "predictions" / "final_tracks.npz",
        prediction_root / scene_id / "final_tracks.npz",
        prediction_root / scene_id / "tracks.npz",
        prediction_root / f"{scene_id}.npz",
    ]
    if allow_direct:
        candidates = [
            prediction_root / "predictions" / prediction_name,
            prediction_root / prediction_name,
            prediction_root / "predictions" / "final_tracks.npz",
            prediction_root / "final_tracks.npz",
            prediction_root / "tracks.npz",
        ] + candidates
    for path in candidates:
        if path.exists():
            return path
    scene_root = prediction_root / scene_id
    if scene_root.exists():
        npz_files = sorted(scene_root.rglob("*.npz"))
        if npz_files:
            return npz_files[-1]
    return None


def _parse_scene_ids(scene_ids):
    scene_ids = scene_ids.replace(",", " ")
    return [item for item in scene_ids.split() if item]


def _read_first_key(npz, keys, default=_MISSING):
    for key in keys:
        if key in npz:
            return npz[key]
    if default is not _MISSING:
        return default
    raise KeyError(f"None of {keys} found in {npz.files}")


def _read_sample_indices(pred, prediction_root, scene_id):
    if "sample_indices" in pred:
        indices = np.asarray(pred["sample_indices"]).reshape(-1)
        if indices.size > 0:
            return indices.astype(np.int64)
    if prediction_root is None:
        return None
    cache_dir = Path(prediction_root) / "point_cache"
    if not cache_dir.exists():
        return None
    matches = sorted(cache_dir.glob(f"{scene_id}_S*_N*_seed*.npz"))
    if not matches:
        return None
    return np.load(matches[-1])["indices"].astype(np.int64)


def _squeeze_batch(array):
    array = np.asarray(array)
    if array.ndim >= 3 and array.shape[0] == 1:
        return array[0]
    return array


def _normalize_prediction_visibility(pred_vis, gt_vis):
    pred_vis = np.asarray(pred_vis)
    gt_vis = np.asarray(gt_vis)
    if pred_vis.ndim == 4 and pred_vis.shape[0] == 1:
        pred_vis = pred_vis[0]
    if pred_vis.ndim == 3 and pred_vis.shape[0] == 1 and pred_vis.shape[1:] == gt_vis.shape[:2]:
        pred_vis = pred_vis[0]
    if pred_vis.ndim == 3 and gt_vis.ndim == 3:
        gt_s, gt_n, gt_v = gt_vis.shape
        if pred_vis.shape == (gt_v, gt_s, gt_n):
            pred_vis = pred_vis.transpose(1, 2, 0)
        elif pred_vis.shape == (gt_s, gt_v, gt_n):
            pred_vis = pred_vis.transpose(0, 2, 1)
    return pred_vis


def _default_query_points(gt_tracks):
    query_points = np.zeros((gt_tracks.shape[1], 4), dtype=np.float32)
    query_points[:, 1:] = gt_tracks[0]
    return query_points


def _align_shapes(pred_tracks, pred_vis, gt_tracks, gt_vis, query_points):
    S = min(pred_tracks.shape[0], gt_tracks.shape[0], gt_vis.shape[0])
    N = min(pred_tracks.shape[1], gt_tracks.shape[1], gt_vis.shape[1], query_points.shape[0])
    pred_tracks = pred_tracks[:S, :N]
    gt_tracks = gt_tracks[:S, :N]
    gt_vis = gt_vis[:S, :N]
    query_points = query_points[:N]

    if pred_vis.ndim == 3:
        pred_vis = pred_vis[:S, :N]
    else:
        pred_vis = pred_vis[:S, :N]
    return pred_tracks, pred_vis, gt_tracks, gt_vis, query_points


def _has_camera(npz):
    return "cam_k" in npz and "cam_rt" in npz


def _evaluate_projection_2d(
    pred_tracks,
    pred_visibility,
    gt_tracks,
    gt_visibility,
    query_points,
    intrinsics,
    extrinsics,
    include_query,
    mvtracker_strict=False,
    scale_metrics_by_100=False,
):
    intrinsics = np.asarray(intrinsics)
    extrinsics = np.asarray(extrinsics)
    S = min(pred_tracks.shape[0], intrinsics.shape[0], extrinsics.shape[0])
    pred_tracks = pred_tracks[:S]
    gt_tracks = gt_tracks[:S]
    gt_visibility = gt_visibility[:S]
    if pred_visibility.ndim == 3:
        pred_visibility = pred_visibility[:S]
    else:
        pred_visibility = pred_visibility[:S]
    intrinsics = intrinsics[:S]
    extrinsics = extrinsics[:S]

    gt_uv, gt_depth = _project_world_to_views(gt_tracks, intrinsics, extrinsics)
    pred_uv, pred_depth = _project_world_to_views(pred_tracks, intrinsics, extrinsics)

    V = gt_uv.shape[0]
    pred_vis_view = _visibility_per_view(pred_visibility, V) & (pred_depth > 1e-6)
    gt_vis_view = _visibility_per_view(gt_visibility, V) & (gt_depth > 1e-6)

    rows = {}
    per_view = []
    for view_idx in range(V):
        metrics = compute_2dpt_metrics(
            pred_tracks=pred_uv[view_idx],
            pred_visibility=pred_vis_view[view_idx],
            gt_tracks=gt_uv[view_idx],
            gt_visibility=gt_vis_view[view_idx],
            query_points=query_points,
            include_query=include_query,
            mvtracker_strict=mvtracker_strict,
            scale_metrics_by_100=scale_metrics_by_100,
        )
        rows[f"view_{view_idx:03d}"] = metrics
        per_view.append(metrics)
    rows["view_mean"] = _mean_metric_dicts(per_view)
    return rows


def _project_world_to_views(world_tracks, intrinsics, extrinsics):
    points_h = np.concatenate(
        [world_tracks.astype(np.float32), np.ones((*world_tracks.shape[:2], 1), dtype=np.float32)],
        axis=-1,
    )
    cam = np.einsum("svij,snj->svni", extrinsics.astype(np.float32), points_h)[..., :3]
    depth = cam[..., 2]
    safe_depth = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    norm = cam / safe_depth[..., None]
    pix = np.einsum("svij,svnj->svni", intrinsics.astype(np.float32), norm)
    uv = pix[..., :2].transpose(1, 0, 2, 3)
    depth = depth.transpose(1, 0, 2)
    return uv, depth


def _visibility_per_view(visibility, num_views):
    visibility = np.asarray(visibility)
    if visibility.ndim == 3:
        return (visibility > 0.5).transpose(2, 0, 1)
    return np.broadcast_to((visibility > 0.5)[None], (num_views,) + visibility.shape)


def _write_csv(path, rows):
    headers = ["scene_id", "prediction"] + METRIC_GROUPS + ["num_eval_points", "num_visible_points"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_group_csv(path, rows, group_key):
    headers = [
        "scene_id",
        "prediction",
        group_key,
    ] + METRIC_GROUPS + ["num_eval_points", "num_visible_points", "num_tracks", "track_percent", "mean_motion"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _average_rows(rows):
    averages = {}
    for key in METRIC_GROUPS + ["num_eval_points", "num_visible_points"]:
        averages[key] = float(np.mean([row[key] for row in rows]))
    averages["num_sequences"] = len(rows)
    return averages


def _average_group_rows(rows, group_key):
    averages = {}
    groups = sorted({row[group_key] for row in rows})
    for group in groups:
        group_rows = [row for row in rows if row[group_key] == group]
        averages[group] = _mean_metric_dicts(group_rows)
        averages[group]["num_sequences"] = len(group_rows)
    return averages


def _mean_metric_dicts(rows):
    if not rows:
        return {}
    keys = [
        key
        for key in METRIC_GROUPS + ["num_eval_points", "num_visible_points", "num_tracks", "track_percent", "mean_motion"]
        if key in rows[0]
    ]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _named_row(scene_id, pred_file, group_key, group, metrics):
    row = {
        "scene_id": scene_id,
        "prediction": str(pred_file),
        group_key: group,
    }
    row.update(metrics)
    return row


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _write_markdown(path, rows, averages):
    headers = ["scene_id", "average_jaccard", "average_pts_within_thresh", "occlusion_accuracy", "mte_visible", "fde_visible", "survival"]
    with open(path, "w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(_format_cell(row[h]) for h in headers) + " |\n")
        f.write("| average | " + " | ".join(_format_cell(averages[h]) for h in headers[1:]) + " |\n")


def _format_cell(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Track3D-TTO predictions.")
    parser.add_argument("--prediction_root", default="outputs")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir", default="outputs/eval")
    parser.add_argument("--setting", default="panoptic-multiview")
    parser.add_argument("--include_query", action="store_true")
    parser.add_argument("--scene_id", default="")
    parser.add_argument("--scene_ids", default="")
    parser.add_argument("--prediction_name", default="final_tracks.npz")
    parser.add_argument("--no_eval_2d_projection", action="store_true")
    parser.add_argument("--mvtracker_strict", action="store_true")
    parser.add_argument("--scale_metrics_by_100", action="store_true")
    args = parser.parse_args()
    main(
        prediction_root=args.prediction_root,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        setting=args.setting,
        include_query=args.include_query,
        scene_id=args.scene_id,
        scene_ids=args.scene_ids,
        prediction_name=args.prediction_name,
        eval_2d_projection=not args.no_eval_2d_projection,
        mvtracker_strict=args.mvtracker_strict,
        scale_metrics_by_100=args.scale_metrics_by_100,
    )
