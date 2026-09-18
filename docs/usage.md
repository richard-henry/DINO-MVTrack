# Using DINO-MVTrack

Run commands from the repository root. The current public interface supports
prepared benchmark scenes with four calibrated views, S24 frames, and N128
point slots. GPU optimization needs compatible initial weights and a frozen
DINOv3-S cache. Dataset and weight downloads are not bundled.

## Prepare inputs

Export scenes in the [documented format](data-format.md). Keep the same point
IDs, camera identities, query convention, and world units throughout.

Create deterministic point caches, then triangulate the matched pseudo tracks:

```bash
python -m track3d.cli.sample_points \
  --dataset_root data/scenes --scene_ids example_scene \
  --output_root data --num_frames 24 --num_queries 128 --seed 125

python -m track3d.cli.triangulate \
  --dataset_root data/scenes --scene_ids example_scene \
  --sample_indices_root data/point_cache \
  --output_root data/triangulation --max_frames 24
```

Use a fresh triangulation output directory to preserve previous predictions.
The sample utility selects query-valid points deterministically; it does not
perform GT-conditioned teacher-quality filtering. This preprocessing example
does not reconstruct an unpublished benchmark point cohort automatically.

Supply `cotracker.npz` from the intended teacher. The optional
`python -m track3d.cli.pseudo_tracks --help` utility generates **CoTracker3
offline** tracks and uses reference visibility to initialize first-visible
queries per view. This is a benchmark helper, not the CoTracker3 online
teacher configuration used in every reported experiment. Record the actual
teacher and query policy when reporting results.

## Configure a scene

Copy `configs/scene.example.json` to `configs/scene.json` and set:

- `dataset_location`, `target_seq`: exported scene root and directory name.
- `sample_cache_dir`: matching point-cache directory.
- `expected_selected_views`: four original camera IDs.
- `triangulation_init_root`: the root containing scene triangulations.
- `semantic_cache_path`: the frozen feature-cache file.
- `common_init_checkpoint`: compatible common model initialization, including
  the `fnet`, `fusion_net`, and `dino_proj` parameter groups.
- `cotracker_encoder_checkpoint`: compatible extracted encoder state dictionary.

Paths are relative to the JSON file, or absolute paths on your machine.
Compatible common initialization and extracted encoder download links are
pending. Arbitrary tracker checkpoints are not interchangeable with these
state dictionaries.

## Prepare frozen DINO features

The RGB bundle is built from the same loader used during optimization:

```bash
python -m track3d.cli.track prepare-rgb \
  --scene-config configs/scene.json --output data/example_rgb.pt
```

This step needs the scene, pseudo tracks, point cache, and triangulation; the
feature cache and model weights need not exist yet.

For the native DINOv3 extractor, use a separate Python >=3.10 environment with
the dependencies of the official DINOv3 checkout. Obtain the authorized ViT-S/16
weights from the original provider. The source commit and checkpoint SHA256
are pinned in `configs/dinov3_native.json`.

```bash
python -m track3d.cli.cache_features \
  --rgb-bundle data/example_rgb.pt --output data/example_scene_dinov3.pt \
  --family dinov3 --native-repo /path/to/dinov3 \
  --model-path /path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

The extractor also supports a Hugging Face model via `--model-path`. That
requires a Transformers version supporting DINOv3 in the separate extraction
environment; the pinned tracking environment's Transformers 4.46.3 does not.
Different extraction backends must not silently be treated as identical caches.

## Check and optimize

```bash
python -m track3d.cli.track check --scene-config configs/scene.json
python -m track3d.cli.track optimize \
  --scene-config configs/scene.json --output outputs/example --device 0
```

`check` loads the real scene and validates frozen features against its RGB
content on CPU. It confirms that weight files exist; compatibility of weight
tensor names and shapes is checked when constructing/loading the model.

`optimize` runs the fixed algorithm in `configs/tracker.json` with the stopping
rule in `configs/convergence.json`. The loss-selected model may precede the
last optimization step. Continuation uses the terminal state, not the selected
earlier weights. `result.json` records selected step, actual updates, stopping
reason, prediction location, and exported arrays.

The output contains the initial run, an optional continuation, logs, a result
record, and `tracks.npz`. Its projected coordinates do not require GT tracks.
To export an existing prediction separately:

```bash
python -m track3d.cli.track export \
  --prediction /path/to/final_tracks.npz \
  --cameras data/scenes/example_scene/track.npz \
  --output outputs/exported_tracks.npz
```

## Evaluate

For a collection arranged as `<prediction_root>/<scene>/predictions/final_tracks.npz`:

```bash
python -m track3d.cli.eval \
  --dataset_root data/scenes --prediction_root outputs/predictions \
  --output_dir outputs/evaluation --setting panoptic-multiview --mvtracker_strict
```

Use the dataset-specific setting exposed by the evaluator. Report point-set
identity, teacher, initialization, input depth/calibration, and model selection.
GT-selected saved-step results are a separate diagnostic, and selected
qualitative cases do not estimate aggregate performance.

## CPU checks

```bash
python -m unittest discover -s tests -v
```

These checks cover input configuration, optimization continuation, and exported
camera projections. They do not replace a complete GPU optimization run or a
clean-machine installation test.
