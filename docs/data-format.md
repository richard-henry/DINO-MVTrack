# Data format

An exported scene has the following layout:

```text
scenes/example_scene/
  track.npz
  cotracker.npz
  0/rgb.mp4
  1/rgb.mp4
  2/rgb.mp4
  3/rgb.mp4
```

Numbered images (`000000.jpg` or `.png`) can replace each `rgb.mp4`. The numbered
folders identify the exported camera slots; `selected_views` records their
original camera IDs. All requested frames must exist and share the input size.

## Scene arrays

`track.npz` contains:

| Key | Shape | Meaning |
|---|---|---|
| `trajs_3d` | S, N, 3 | Reference world trajectories for benchmark metadata/evaluation |
| `visibility` | S, N, V | Reference per-view visibility |
| `cam_k` | S, V, 3, 3 | Camera intrinsics |
| `cam_rt` | S, V, 4, 4 | World-to-camera transforms |
| `query_points` | N, 4 | Task queries `(frame, x, y, z)` |
| `selected_views` | V | Original camera IDs, matching the scene configuration |

Supply explicit task queries. The benchmark loader has a historical fallback
that derives queries from the first reference frame; this must not be described
as a GT-free input protocol. Coordinates, cameras, and initialization must use
the same world units and coordinate frame.

`cotracker.npz` contains `pred_tracks [V,S,N,2]` in original image pixels and
`pred_visibility [V,S,N]`. Preserve point identity across all files.

Point caches contain integer `indices` and are named
`<scene>_S24_N128_seed125.npz`. Triangulation is stored under
`<triangulation_root>/<scene>/predictions/final_tracks.npz`; the triangulation
utility generates coordinates, visibility, quality fields, and `sample_indices`.

The frozen DINO feature cache validates the exact RGB tensor, frame order,
view order, feature geometry, and encoder identity. It is not transferable to
a different clip merely because tensor dimensions match.

## Output arrays

The optimization entry exports `tracks.npz`:

| Key | Shape | Meaning |
|---|---|---|
| `tracks_3d` | S, N, 3 | World-coordinate trajectories |
| `tracks_2d` | S, V, N, 2 | Camera projections in original pixels |
| `visibility_any_view` | S, N | Triangulation-derived any-view visibility |
| `projection_valid` | S, V, N | Finite projection with positive camera depth |
| `query_active` | S, N | Whether the task query has occurred |
| `sample_indices` | N | Original point IDs |
| `query_points` | N, 4 | Task query identities |
| `selected_views` | V | Camera IDs |

Invalid projections contain NaN. `projection_valid` does not mean that a point
is inside the image or unoccluded. Any-view visibility is not per-view
occlusion prediction. Saved optimization step and visibility source accompany
the arrays; model step 0 is distinct from triangulation initialization.
