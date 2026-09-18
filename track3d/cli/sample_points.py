"""Create deterministic Track3D point-cache files for benchmark subsets."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--scene_ids", default="")
    parser.add_argument("--num_frames", type=int, default=24)
    parser.add_argument("--num_queries", type=int, default=128)
    parser.add_argument("--seed", type=int, default=125)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    cache_dir = output_root / "point_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    scene_ids = _parse_scene_ids(args.scene_ids)
    scene_dirs = [dataset_root / sid for sid in scene_ids] if scene_ids else sorted(
        p for p in dataset_root.iterdir() if p.is_dir()
    )

    rows = []
    for scene_dir in scene_dirs:
        track_path = scene_dir / "track.npz"
        if not track_path.exists():
            print(f"[skip] missing {track_path}")
            continue
        data = np.load(track_path, allow_pickle=True)
        num_tracks = int(data["trajs_3d"].shape[1])
        query_points = data["query_points"] if "query_points" in data else _default_query_points(data["trajs_3d"])
        use_s = min(int(args.num_frames), int(data["trajs_3d"].shape[0]))
        valid_indices = np.where(query_points[:, 0] < use_s)[0]
        if len(valid_indices) == 0:
            print(f"[warn] {scene_dir.name}: no query points before S={use_s}; using all tracks")
            valid_indices = np.arange(num_tracks)

        rng = np.random.default_rng(_sequence_seed(args.seed, scene_dir.name, use_s, len(valid_indices), args.num_queries))
        if len(valid_indices) > args.num_queries:
            indices = rng.choice(valid_indices, args.num_queries, replace=False)
        elif len(valid_indices) < args.num_queries:
            pad = rng.choice(valid_indices, args.num_queries - len(valid_indices), replace=True)
            indices = np.concatenate([valid_indices, pad])
        else:
            indices = valid_indices
        indices = indices.astype(np.int64)

        out_path = cache_dir / f"{scene_dir.name}_S{use_s}_N{args.num_queries}_seed{args.seed}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {scene_dir.name}: {out_path} exists")
        else:
            np.savez_compressed(out_path, indices=indices)
            print(f"[ok] {scene_dir.name}: valid={len(valid_indices)}, wrote {out_path}")
        rows.append((scene_dir.name, use_s, len(valid_indices), len(indices)))

    if not rows:
        raise RuntimeError(f"No sample caches written from {dataset_root}")


def _parse_scene_ids(scene_ids: str) -> list[str]:
    return [item for item in scene_ids.replace(",", " ").split() if item]


def _default_query_points(tracks: np.ndarray) -> np.ndarray:
    query_points = np.zeros((tracks.shape[1], 4), dtype=np.float32)
    query_points[:, 1:] = tracks[0]
    return query_points


def _sequence_seed(seed: int, scene_name: str, use_s: int, valid_count: int, num_queries: int) -> int:
    key = f"{seed}:{scene_name}:S{use_s}:N{num_queries}:valid{valid_count}"
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)


if __name__ == "__main__":
    main()
