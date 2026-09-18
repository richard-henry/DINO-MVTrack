"""Triangulate multi-view 2D pseudo tracks into 3D trajectory predictions."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--scene_ids", default="")
    parser.add_argument("--pseudo_name", default="cotracker.npz")
    parser.add_argument("--prediction_name", default="final_tracks.npz")
    parser.add_argument("--sample_indices_root", default="")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--max_reproj_error", type=float, default=6.0)
    parser.add_argument("--max_view_error", type=float, default=12.0)
    parser.add_argument("--min_ray_angle_deg", type=float, default=1.0)
    parser.add_argument("--edge_margin", type=float, default=0.0)
    parser.add_argument(
        "--image_width",
        type=float,
        default=0.0,
        help="Image width for in-frame filtering. Use <=0 to infer from meta.json or exported images.",
    )
    parser.add_argument(
        "--image_height",
        type=float,
        default=0.0,
        help="Image height for in-frame filtering. Use <=0 to infer from meta.json or exported images.",
    )
    parser.add_argument("--fallback", choices=("query", "interp_query"), default="interp_query")
    parser.add_argument("--point_size_min", type=float, default=0.5)
    parser.add_argument("--point_size_max", type=float, default=2.0)
    parser.add_argument("--point_size_safety", type=float, default=1.5)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    sample_root = Path(args.sample_indices_root) if args.sample_indices_root else None
    scene_ids = _parse_scene_ids(args.scene_ids)
    scene_dirs = [dataset_root / sid for sid in scene_ids] if scene_ids else sorted(p for p in dataset_root.iterdir() if p.is_dir())

    summary_rows = []
    for scene_dir in scene_dirs:
        if not scene_dir.exists():
            print(f"[skip] scene not found: {scene_dir}")
            continue
        sample_indices = _read_sample_indices(sample_root, scene_dir.name) if sample_root else None
        image_width, image_height = _resolve_image_size(scene_dir, args.image_width, args.image_height)
        result = triangulate_scene(
            scene_dir=scene_dir,
            pseudo_name=args.pseudo_name,
            sample_indices=sample_indices,
            max_frames=args.max_frames,
            min_views=args.min_views,
            max_reproj_error=args.max_reproj_error,
            max_view_error=args.max_view_error,
            min_ray_angle_deg=args.min_ray_angle_deg,
            edge_margin=args.edge_margin,
            image_width=image_width,
            image_height=image_height,
            fallback=args.fallback,
            point_size_min=args.point_size_min,
            point_size_max=args.point_size_max,
            point_size_safety=args.point_size_safety,
        )
        out_dir = output_root / scene_dir.name / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / args.prediction_name
        np.savez_compressed(
            out_path,
            pred_tracks_3d=result["tracks"].astype(np.float32),
            pred_trajectories_3d=result["tracks"].astype(np.float32),
            pred_visibility=result["visibility"].astype(np.float32),
            pred_visibilities_any_view=result["visibility"].astype(np.float32),
            triangulation_confidence=result["confidence"].astype(np.float32),
            reprojection_error_px=result["reproj_error"].astype(np.float32),
            num_views_used=result["num_views"].astype(np.int16),
            ray_angle_deg=result["ray_angle"].astype(np.float32),
            raw_triangulated_mask=result["raw_valid"].astype(np.bool_),
            raw_tracks_3d=result["raw_tracks"].astype(np.float32),
            raw_confidence=result["raw_confidence"].astype(np.float32),
            raw_reprojection_error_px=result["raw_reproj_error"].astype(np.float32),
            raw_ray_angle_deg=result["raw_ray_angle"].astype(np.float32),
            metadata_version=np.array("raw_observations_v1"),
            fallback=np.array(args.fallback),
            init_size=result["init_size"].astype(np.float32),
            point_size=result["point_size"].astype(np.float32),
            image_width=np.array(image_width, dtype=np.float32),
            image_height=np.array(image_height, dtype=np.float32),
            sample_indices=np.asarray(sample_indices if sample_indices is not None else np.arange(result["tracks"].shape[1]), dtype=np.int64),
            method=np.array("cotracker3_triangulation"),
            input_protocol=np.array("rgb_camera_cotracker3_multiview_triangulation"),
        )
        row = result["summary"]
        row["scene_id"] = scene_dir.name
        row["prediction"] = str(out_path)
        summary_rows.append(row)
        print(
            f"[ok] {scene_dir.name}: valid={row['raw_valid_ratio']:.3f}, "
            f"conf={row['mean_confidence']:.3f}, reproj={row['mean_reproj_error_px']:.3f}px, "
            f"size={int(image_width)}x{int(image_height)} -> {out_path}"
        )

    if not summary_rows:
        raise RuntimeError(f"No scenes triangulated from {dataset_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "triangulation_summary.csv", summary_rows)


def triangulate_scene(
    scene_dir: Path,
    pseudo_name: str,
    sample_indices: np.ndarray | None,
    max_frames: int,
    min_views: int,
    max_reproj_error: float,
    max_view_error: float,
    min_ray_angle_deg: float,
    edge_margin: float,
    image_width: float,
    image_height: float,
    fallback: str,
    point_size_min: float,
    point_size_max: float,
    point_size_safety: float,
) -> dict:
    gt = np.load(scene_dir / "track.npz", allow_pickle=True)
    pseudo = np.load(scene_dir / pseudo_name, allow_pickle=True)

    gt_tracks = gt["trajs_3d"].astype(np.float32)
    query_points = gt["query_points"].astype(np.float32) if "query_points" in gt else _default_query_points(gt_tracks)
    cam_k = gt["cam_k"].astype(np.float32)
    cam_rt = gt["cam_rt"].astype(np.float32)
    pseudo_tracks = pseudo["pred_tracks"].astype(np.float32)
    pseudo_vis = pseudo["pred_visibility"].astype(np.float32) > 0.5

    if max_frames > 0:
        S = min(max_frames, gt_tracks.shape[0], cam_k.shape[0], pseudo_tracks.shape[1])
    else:
        S = min(gt_tracks.shape[0], cam_k.shape[0], pseudo_tracks.shape[1])
    if sample_indices is None:
        sample_indices = np.arange(gt_tracks.shape[1], dtype=np.int64)
    else:
        sample_indices = np.asarray(sample_indices).reshape(-1).astype(np.int64)

    gt_tracks = gt_tracks[:S, sample_indices]
    query_points = query_points[sample_indices]
    cam_k = cam_k[:S]
    cam_rt = cam_rt[:S]
    pseudo_tracks = pseudo_tracks[:, :S, sample_indices]
    pseudo_vis = pseudo_vis[:, :S, sample_indices]

    V, _, N, _ = pseudo_tracks.shape
    projection_mats = np.einsum("svij,svjk->svik", cam_k, cam_rt[:, :, :3, :])
    cam_centers = _camera_centers(cam_rt)
    tracks = np.zeros((S, N, 3), dtype=np.float32)
    confidence = np.zeros((S, N), dtype=np.float32)
    reproj_error = np.full((S, N), np.nan, dtype=np.float32)
    num_views = np.zeros((S, N), dtype=np.int16)
    ray_angle = np.zeros((S, N), dtype=np.float32)
    raw_valid = np.zeros((S, N), dtype=bool)

    for n in range(N):
        q_t = int(np.clip(round(float(query_points[n, 0])), 0, S - 1))
        for s in range(S):
            if s < q_t:
                continue
            valid_views = _valid_observation_views(
                pseudo_tracks[:, s, n],
                pseudo_vis[:, s, n],
                image_width=image_width,
                image_height=image_height,
                edge_margin=edge_margin,
            )
            if len(valid_views) < min_views:
                continue
            tri = _robust_triangulate(
                uv=pseudo_tracks[:, s, n],
                view_indices=valid_views,
                projection_mats=projection_mats[s],
                cam_rt=cam_rt[s],
                cam_centers=cam_centers[s],
                min_views=min_views,
                max_reproj_error=max_reproj_error,
                max_view_error=max_view_error,
                min_ray_angle_deg=min_ray_angle_deg,
            )
            if tri is None:
                continue
            tracks[s, n] = tri["point"]
            confidence[s, n] = tri["confidence"]
            reproj_error[s, n] = tri["median_reproj"]
            num_views[s, n] = tri["num_views"]
            ray_angle[s, n] = tri["ray_angle"]
            raw_valid[s, n] = True

    raw_tracks = np.where(raw_valid[..., None], tracks, np.nan).copy()
    raw_confidence = confidence.copy()
    raw_reproj_error = reproj_error.copy()
    raw_ray_angle = ray_angle.copy()
    tracks, visibility = _fill_missing_tracks(
        tracks=tracks,
        raw_valid=raw_valid,
        query_points=query_points,
        fallback=fallback,
    )
    if fallback == "interp_query":
        confidence = _fill_missing_scalar(confidence, raw_valid, fill_value=0.0)
        reproj_error = _fill_missing_scalar(reproj_error, raw_valid, fill_value=np.nan)
        ray_angle = _fill_missing_scalar(ray_angle, raw_valid, fill_value=0.0)
    point_size = _estimate_point_sizes(
        tracks=tracks,
        raw_valid=raw_valid,
        confidence=confidence,
        reproj_error=reproj_error,
        query_points=query_points,
        min_size=point_size_min,
        max_size=point_size_max,
        safety=point_size_safety,
    )
    init_size = np.repeat(point_size[:, None], 3, axis=1).astype(np.float32)

    valid_values = raw_valid.sum()
    summary = {
        "num_frames": int(S),
        "num_tracks": int(N),
        "image_width": float(image_width),
        "image_height": float(image_height),
        "raw_valid_ratio": float(raw_valid.mean()),
        "raw_valid_points": int(valid_values),
        "mean_confidence": float(np.nanmean(confidence[raw_valid])) if valid_values else 0.0,
        "mean_reproj_error_px": float(np.nanmean(reproj_error[raw_valid])) if valid_values else np.nan,
        "median_reproj_error_px": float(np.nanmedian(reproj_error[raw_valid])) if valid_values else np.nan,
        "mean_num_views": float(np.mean(num_views[raw_valid])) if valid_values else 0.0,
        "mean_ray_angle_deg": float(np.mean(ray_angle[raw_valid])) if valid_values else 0.0,
        "filled_visibility_ratio": float(visibility.mean()),
        "mean_point_size": float(np.mean(point_size)),
        "min_point_size": float(np.min(point_size)),
        "max_point_size": float(np.max(point_size)),
    }
    return {
        "tracks": tracks,
        "visibility": visibility,
        "confidence": confidence,
        "reproj_error": reproj_error,
        "num_views": num_views,
        "ray_angle": ray_angle,
        "raw_valid": raw_valid,
        "raw_tracks": raw_tracks,
        "raw_confidence": raw_confidence,
        "raw_reproj_error": raw_reproj_error,
        "raw_ray_angle": raw_ray_angle,
        "point_size": point_size,
        "init_size": init_size,
        "summary": summary,
    }


def _resolve_image_size(scene_dir: Path, image_width: float, image_height: float) -> tuple[float, float]:
    if image_width > 0 and image_height > 0:
        return float(image_width), float(image_height)

    inferred = _infer_image_size(scene_dir)
    if inferred is None:
        fallback = (640.0, 360.0)
        width = float(image_width) if image_width > 0 else fallback[0]
        height = float(image_height) if image_height > 0 else fallback[1]
        print(
            f"[warn] {scene_dir.name}: could not infer image size; "
            f"falling back to {int(width)}x{int(height)}"
        )
        return width, height

    inferred_width, inferred_height = inferred
    width = float(image_width) if image_width > 0 else float(inferred_width)
    height = float(image_height) if image_height > 0 else float(inferred_height)
    return width, height


def _infer_image_size(scene_dir: Path) -> tuple[int, int] | None:
    meta_path = scene_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        width = meta.get("width")
        height = meta.get("height")
        if width and height:
            return int(width), int(height)

    for view_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
        for pattern in ("*.png", "*.jpg", "*.jpeg"):
            first = next(iter(sorted(view_dir.glob(pattern))), None)
            if first is None:
                continue
            image = cv2.imread(str(first), cv2.IMREAD_UNCHANGED)
            if image is not None:
                height, width = image.shape[:2]
                return int(width), int(height)
    return None


def _robust_triangulate(
    uv: np.ndarray,
    view_indices: list[int],
    projection_mats: np.ndarray,
    cam_rt: np.ndarray,
    cam_centers: np.ndarray,
    min_views: int,
    max_reproj_error: float,
    max_view_error: float,
    min_ray_angle_deg: float,
) -> dict | None:
    best = None
    max_subset = min(len(view_indices), 4)
    for subset_size in range(min_views, max_subset + 1):
        for subset in combinations(view_indices, subset_size):
            point = _linear_triangulate(uv, subset, projection_mats)
            if point is None or not np.isfinite(point).all():
                continue
            depths = _camera_depths(point, cam_rt[list(subset)])
            if np.count_nonzero(depths > 1e-6) < min_views:
                continue
            reproj = _reprojection_errors(point, uv, subset, projection_mats)
            if not np.isfinite(reproj).all():
                continue
            median_reproj = float(np.median(reproj))
            max_reproj = float(np.max(reproj))
            if median_reproj > max_reproj_error or max_reproj > max_view_error:
                continue
            angle = _max_ray_angle(point, cam_centers[list(subset)])
            if angle < min_ray_angle_deg:
                continue
            score = _triangulation_score(
                num_views=len(subset),
                median_reproj=median_reproj,
                max_reproj=max_reproj,
                ray_angle_deg=angle,
            )
            if best is None or score > best["score"]:
                best = {
                    "point": point.astype(np.float32),
                    "score": score,
                    "confidence": score,
                    "median_reproj": median_reproj,
                    "max_reproj": max_reproj,
                    "num_views": len(subset),
                    "ray_angle": angle,
                }
    return best


def _linear_triangulate(uv: np.ndarray, view_indices: tuple[int, ...], projection_mats: np.ndarray) -> np.ndarray | None:
    rows = []
    for view_idx in view_indices:
        u, v = uv[view_idx]
        p = projection_mats[view_idx]
        rows.append(u * p[2] - p[0])
        rows.append(v * p[2] - p[1])
    a = np.stack(rows, axis=0).astype(np.float64)
    try:
        _, _, vh = np.linalg.svd(a)
    except np.linalg.LinAlgError:
        return None
    x_h = vh[-1]
    if abs(x_h[-1]) < 1e-10:
        return None
    return (x_h[:3] / x_h[-1]).astype(np.float32)


def _reprojection_errors(point: np.ndarray, uv: np.ndarray, view_indices: tuple[int, ...], projection_mats: np.ndarray) -> np.ndarray:
    x_h = np.concatenate([point.astype(np.float64), np.ones(1, dtype=np.float64)])
    errors = []
    for view_idx in view_indices:
        pix_h = projection_mats[view_idx].astype(np.float64) @ x_h
        if abs(pix_h[2]) < 1e-10:
            errors.append(np.inf)
            continue
        pix = pix_h[:2] / pix_h[2]
        errors.append(float(np.linalg.norm(pix - uv[view_idx])))
    return np.asarray(errors, dtype=np.float32)


def _camera_depths(point: np.ndarray, cam_rt_subset: np.ndarray) -> np.ndarray:
    x_h = np.concatenate([point.astype(np.float64), np.ones(1, dtype=np.float64)])
    cam = np.einsum("vij,j->vi", cam_rt_subset.astype(np.float64), x_h)
    return cam[:, 2]


def _camera_centers(cam_rt: np.ndarray) -> np.ndarray:
    rotations = cam_rt[:, :, :3, :3]
    translations = cam_rt[:, :, :3, 3]
    return -np.einsum("svij,svj->svi", np.swapaxes(rotations, -1, -2), translations)


def _max_ray_angle(point: np.ndarray, centers: np.ndarray) -> float:
    if centers.shape[0] < 2:
        return 0.0
    rays = point[None] - centers
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    rays = rays / np.maximum(norms, 1e-8)
    max_angle = 0.0
    for i, j in combinations(range(centers.shape[0]), 2):
        cosine = float(np.clip(np.dot(rays[i], rays[j]), -1.0, 1.0))
        max_angle = max(max_angle, float(np.degrees(np.arccos(cosine))))
    return max_angle


def _triangulation_score(num_views: int, median_reproj: float, max_reproj: float, ray_angle_deg: float) -> float:
    view_score = np.clip((num_views - 1) / 3.0, 0.0, 1.0)
    reproj_score = float(np.exp(-median_reproj / 3.0))
    max_score = float(np.exp(-max_reproj / 8.0))
    geometry_score = np.clip(ray_angle_deg / 5.0, 0.0, 1.0)
    return float(view_score * reproj_score * max_score * geometry_score)


def _valid_observation_views(
    uv: np.ndarray,
    visibility: np.ndarray,
    image_width: float,
    image_height: float,
    edge_margin: float,
) -> list[int]:
    valid = []
    for view_idx, ((x, y), is_visible) in enumerate(zip(uv, visibility)):
        if not is_visible or not np.isfinite([x, y]).all():
            continue
        if x < edge_margin or y < edge_margin:
            continue
        if x >= image_width - edge_margin or y >= image_height - edge_margin:
            continue
        valid.append(view_idx)
    return valid


def _fill_missing_tracks(
    tracks: np.ndarray,
    raw_valid: np.ndarray,
    query_points: np.ndarray,
    fallback: str,
) -> tuple[np.ndarray, np.ndarray]:
    filled = tracks.copy()
    S, N, _ = filled.shape
    visibility = raw_valid.astype(np.float32)
    for n in range(N):
        q_t = int(np.clip(round(float(query_points[n, 0])), 0, S - 1))
        valid_frames = np.flatnonzero(raw_valid[:, n] & (np.arange(S) >= q_t))
        if fallback == "interp_query" and valid_frames.size > 0:
            for dim in range(3):
                filled[q_t:, n, dim] = np.interp(
                    np.arange(q_t, S),
                    valid_frames,
                    filled[valid_frames, n, dim],
                    left=filled[valid_frames[0], n, dim],
                    right=filled[valid_frames[-1], n, dim],
                )
            visibility[q_t:, n] = 1.0
        else:
            filled[q_t:, n] = query_points[n, 1:].astype(np.float32)
            visibility[q_t:, n] = 1.0
        if q_t > 0:
            filled[:q_t, n] = query_points[n, 1:].astype(np.float32)
            visibility[:q_t, n] = 0.0
    return filled, visibility


def _fill_missing_scalar(values: np.ndarray, raw_valid: np.ndarray, fill_value: float) -> np.ndarray:
    filled = values.copy()
    S, N = filled.shape
    for n in range(N):
        valid_frames = np.flatnonzero(raw_valid[:, n])
        if valid_frames.size == 0:
            filled[:, n] = fill_value
            continue
        finite_values = np.nan_to_num(filled[valid_frames, n], nan=fill_value)
        filled[:, n] = np.interp(np.arange(S), valid_frames, finite_values, left=finite_values[0], right=finite_values[-1])
    return filled


def _estimate_point_sizes(
    tracks: np.ndarray,
    raw_valid: np.ndarray,
    confidence: np.ndarray,
    reproj_error: np.ndarray,
    query_points: np.ndarray,
    min_size: float,
    max_size: float,
    safety: float,
) -> np.ndarray:
    """Estimate a per-track search cube edge length from triangulation quality.

    The value is an edge length, matching `create_boundingbox`, so the actual
    per-axis search radius is `point_size / 2`.
    """
    S, N, _ = tracks.shape
    point_sizes = np.zeros((N,), dtype=np.float32)
    frame_ids = np.arange(S)
    for n in range(N):
        q_t = int(np.clip(round(float(query_points[n, 0])), 0, S - 1))
        post = frame_ids > q_t
        post_count = max(int(post.sum()), 1)
        raw_post = raw_valid[:, n] & post
        raw_ratio = float(raw_post.sum() / post_count)
        valid_conf = confidence[raw_post, n]
        mean_conf = float(np.nanmean(valid_conf)) if valid_conf.size else 0.0
        valid_reproj = reproj_error[raw_post, n]
        reproj_p90 = float(np.nanpercentile(valid_reproj, 90)) if valid_reproj.size else 6.0
        max_gap = _max_missing_gap(raw_post, q_t, S)

        post_track = tracks[post, n]
        if len(post_track) >= 2:
            step = np.linalg.norm(post_track[1:] - post_track[:-1], axis=-1)
            step_p90 = float(np.nanpercentile(step, 90))
        else:
            step_p90 = 0.0

        # Convert quality proxies into a world-space radius. This deliberately
        # overestimates uncertain/interpolated tracks and keeps high-confidence
        # raw triangulated tracks tight.
        quality_penalty = 0.08 * (1.0 - raw_ratio) + 0.06 * (1.0 - mean_conf)
        reproj_penalty = 0.012 * min(reproj_p90, 12.0)
        gap_penalty = 0.012 * min(max_gap, 12)
        motion_penalty = 0.50 * min(step_p90, 0.75)
        radius = safety * (0.10 + quality_penalty + reproj_penalty + gap_penalty + motion_penalty)
        point_sizes[n] = np.clip(2.0 * radius, min_size, max_size)
    return point_sizes.astype(np.float32)


def _max_missing_gap(raw_post: np.ndarray, query_t: int, num_frames: int) -> int:
    post_frames = np.arange(query_t + 1, num_frames)
    if len(post_frames) == 0:
        return 0
    raw_frames = np.flatnonzero(raw_post)
    if raw_frames.size == 0:
        return len(post_frames)
    anchors = np.concatenate([[query_t], raw_frames, [num_frames - 1]])
    return int(np.max(np.diff(np.unique(anchors))))


def _read_sample_indices(sample_root: Path, scene_id: str) -> np.ndarray:
    candidates = [
        sample_root / scene_id / "predictions" / "final_tracks.npz",
        sample_root / scene_id / "final_tracks.npz",
    ]
    for path in candidates:
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        if "sample_indices" in data:
            indices = np.asarray(data["sample_indices"]).reshape(-1)
            if indices.size:
                return indices.astype(np.int64)
    matches = sorted((sample_root / scene_id / "point_cache").glob(f"{scene_id}_S*_N*_seed*.npz"))
    if matches:
        return np.load(matches[-1])["indices"].astype(np.int64)
    matches = sorted((sample_root / "point_cache").glob(f"{scene_id}_S*_N*_seed*.npz"))
    if matches:
        return np.load(matches[-1])["indices"].astype(np.int64)
    matches = sorted(sample_root.glob(f"{scene_id}_S*_N*_seed*.npz"))
    if matches:
        return np.load(matches[-1])["indices"].astype(np.int64)
    raise FileNotFoundError(f"No sample_indices found for {scene_id} under {sample_root}")


def _default_query_points(gt_tracks: np.ndarray) -> np.ndarray:
    query_points = np.zeros((gt_tracks.shape[1], 4), dtype=np.float32)
    query_points[:, 1:] = gt_tracks[0]
    return query_points


def _parse_scene_ids(scene_ids: str) -> list[str]:
    return [item for item in scene_ids.replace(",", " ").split() if item]


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["scene_id", "prediction"] + [key for key in rows[0].keys() if key not in {"scene_id", "prediction"}]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
