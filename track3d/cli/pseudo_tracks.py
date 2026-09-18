"""Export 2D tracker pseudo labels for Track3D benchmark scenes."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


def main(
    tracker: str = "cotracker3",
    dataset_root: str = "",
    output_name: str = "cotracker.npz",
    scene_id: str = "",
    views: str = "",
    checkpoint: str = "",
    hub_repo: str = "",
    device: str = "auto",
    backward_tracking: bool = True,
    query_policy: str = "first_visible_per_view",
    max_query_chunk: int = 512,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    if tracker != "cotracker3":
        raise ValueError("Only --tracker cotracker3 is implemented")
    if not dataset_root:
        raise ValueError("--dataset_root is required")

    root = Path(dataset_root)
    scene_dirs = [root / scene_id] if scene_id else sorted(p for p in root.iterdir() if p.is_dir())
    selected_views = _parse_views(views)

    predictor = None
    device = _resolve_device(device)
    if not dry_run:
        predictor = _load_cotracker3(checkpoint=checkpoint, hub_repo=hub_repo, device=device)

    for scene_dir in scene_dirs:
        if not scene_dir.exists():
            print(f"[skip] missing scene {scene_dir}")
            continue
        out_path = scene_dir / output_name
        if out_path.exists() and not overwrite:
            print(f"[skip] {scene_dir.name}: {out_path.name} exists")
            continue
        info = _load_scene_info(scene_dir, selected_views)
        print(
            f"[scene] {scene_dir.name}: V={len(info['views'])}, S={info['tracks_3d'].shape[0]}, "
            f"N={info['tracks_3d'].shape[1]}, H={info['height']}, W={info['width']}"
        )
        if dry_run:
            continue
        pred_tracks, pred_visibility = _run_cotracker_scene(
            predictor=predictor,
            info=info,
            device=device,
            backward_tracking=backward_tracking,
            query_policy=query_policy,
            max_query_chunk=max_query_chunk,
        )
        np.savez_compressed(
            out_path,
            pred_tracks=pred_tracks.astype(np.float32),
            pred_visibility=pred_visibility.astype(np.float32),
            valid_views=np.array(info["views"], dtype=np.int64),
            tracker_name=np.array("cotracker3_offline"),
            query_policy=np.array(query_policy),
        )
        print(f"[ok] wrote {out_path}")


def _load_cotracker3(checkpoint: str, hub_repo: str, device: str):
    if hub_repo:
        predictor = torch.hub.load(hub_repo, "cotracker3_offline", source="local", pretrained=True)
    elif checkpoint:
        from cotracker.predictor import CoTrackerPredictor

        predictor = CoTrackerPredictor(checkpoint=checkpoint, offline=True, v2=False, window_len=60)
    else:
        cache_repo = Path.home() / ".cache/torch/hub/facebookresearch_co-tracker_main"
        if cache_repo.exists():
            predictor = torch.hub.load(str(cache_repo), "cotracker3_offline", source="local", pretrained=True)
        else:
            predictor = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline", pretrained=True)
    predictor = predictor.to(device)
    predictor.eval()
    return predictor


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] requested {device}, but CUDA is unavailable; falling back to CPU")
        return "cpu"
    return device


def _load_scene_info(scene_dir: Path, selected_views: list[int] | None) -> dict:
    track = np.load(scene_dir / "track.npz", allow_pickle=True)
    tracks_3d = track["trajs_3d"].astype(np.float32)
    visibility = track["visibility"].astype(bool)
    intrinsics = track["cam_k"].astype(np.float32)
    extrinsics = track["cam_rt"].astype(np.float32)
    all_views = list(range(visibility.shape[2]))
    views = selected_views or all_views
    videos = [_read_scene_video(scene_dir / str(view), tracks_3d.shape[0]) for view in views]
    height, width = videos[0].shape[1:3]
    return {
        "views": views,
        "videos": videos,
        "tracks_3d": tracks_3d,
        "visibility": visibility[:, :, views],
        "intrinsics": intrinsics[:, views],
        "extrinsics": extrinsics[:, views],
        "height": height,
        "width": width,
    }


def _run_cotracker_scene(
    predictor,
    info: dict,
    device: str,
    backward_tracking: bool,
    query_policy: str,
    max_query_chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    videos = info["videos"]
    S, N = info["tracks_3d"].shape[:2]
    V = len(videos)
    pred_tracks = np.zeros((V, S, N, 2), dtype=np.float32)
    pred_visibility = np.zeros((V, S, N), dtype=bool)
    query_uv, query_valid = _make_view_queries(info, query_policy=query_policy)

    for view_idx, video_np in enumerate(videos):
        video = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float().to(device)
        for start in range(0, N, max_query_chunk):
            end = min(start + max_query_chunk, N)
            queries_np = query_uv[view_idx, start:end]
            valid_np = query_valid[view_idx, start:end]
            if not valid_np.any():
                continue
            queries = torch.from_numpy(queries_np[valid_np])[None].float().to(device)
            with torch.no_grad():
                tracks, vis = predictor(
                    video,
                    queries=queries,
                    backward_tracking=backward_tracking,
                )
            tracks_np = tracks[0].detach().cpu().numpy().astype(np.float32)
            vis_np = vis[0].detach().cpu().numpy().astype(bool)
            indices = np.nonzero(valid_np)[0] + start
            pred_tracks[view_idx, :, indices, :] = np.transpose(tracks_np, (1, 0, 2))
            pred_visibility[view_idx, :, indices] = vis_np.T
        print(f"  [view {info['views'][view_idx]}] tracked {int(query_valid[view_idx].sum())}/{N} points")
    return pred_tracks, pred_visibility


def _make_view_queries(info: dict, query_policy: str) -> tuple[np.ndarray, np.ndarray]:
    if query_policy != "first_visible_per_view":
        raise ValueError(f"Unsupported query_policy: {query_policy}")
    visibility = info["visibility"]
    S, N, V = visibility.shape
    uv, depth = _project_world_to_views(info["tracks_3d"], info["intrinsics"], info["extrinsics"])
    queries = np.zeros((V, N, 3), dtype=np.float32)
    valid = np.zeros((V, N), dtype=bool)
    for view_idx in range(V):
        view_vis = visibility[:, :, view_idx] & (depth[view_idx] > 1e-6)
        valid[view_idx] = view_vis.any(axis=0)
        first_t = np.argmax(view_vis.astype(np.float32), axis=0)
        queries[view_idx, :, 0] = first_t
        queries[view_idx, :, 1:] = uv[view_idx, first_t, np.arange(N)]
        queries[view_idx, :, 1] = np.clip(queries[view_idx, :, 1], 0, info["width"] - 1)
        queries[view_idx, :, 2] = np.clip(queries[view_idx, :, 2], 0, info["height"] - 1)
    return queries, valid


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
    return pix[..., :2].transpose(1, 0, 2, 3), depth.transpose(1, 0, 2)


def _read_scene_video(view_dir: Path, num_frames: int) -> np.ndarray:
    mp4_path = view_dir / "rgb.mp4"
    if mp4_path.exists():
        frames = _read_mp4(mp4_path, num_frames)
    else:
        paths = sorted(view_dir.glob("*.png")) + sorted(view_dir.glob("*.jpg"))
        frames = [_read_rgb_image(path) for path in paths[:num_frames]]
    if not frames:
        raise FileNotFoundError(f"No frames found in {view_dir}")
    if len(frames) < num_frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))
    return np.stack(frames, axis=0)


def _read_mp4(path: Path, num_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < num_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _read_rgb_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _parse_views(views: str) -> list[int] | None:
    if not views:
        return None
    return [int(item) for item in views.split(",") if item.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", default="cotracker3")
    parser.add_argument("--dataset_root", default="")
    parser.add_argument("--output_name", default="cotracker.npz")
    parser.add_argument("--scene_id", default="")
    parser.add_argument("--views", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--hub_repo", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backward_tracking", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--query_policy", default="first_visible_per_view")
    parser.add_argument("--max_query_chunk", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    main(**vars(args))
