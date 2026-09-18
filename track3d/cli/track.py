"""Validate inputs, optimize a scene, and export DINO-MVTrack predictions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PATH_KEYS = ('dataset_location', 'sample_cache_dir', 'triangulation_init_root',
             'semantic_cache_path', 'common_init_checkpoint', 'cotracker_encoder_checkpoint')
SCENE_KEYS = set(PATH_KEYS) | {'target_seq', 'expected_selected_views'}


def load_scene(path, require_features=True):
    path = Path(path).resolve()
    scene = json.loads(path.read_text())
    if set(scene) != SCENE_KEYS:
        raise ValueError('Scene config must contain exactly: ' + ', '.join(sorted(SCENE_KEYS)))
    name = scene['target_seq']
    if not isinstance(name, str) or name in ('', '.', '..') or Path(name).name != name:
        raise ValueError('target_seq must be a single scene directory name')
    for key in PATH_KEYS:
        if not isinstance(scene[key], str) or not scene[key]:
            raise ValueError('Missing path: ' + key)
        value = Path(scene[key]).expanduser()
        scene[key] = str((path.parent / value).resolve() if not value.is_absolute() else value.resolve())
    views = scene['expected_selected_views']
    if (not isinstance(views, list) or len(views) != 4 or len(set(views)) != 4
            or any(type(v) is not int or v < 0 for v in views)):
        raise ValueError('expected_selected_views must list four distinct nonnegative camera IDs')
    base = Path(scene['dataset_location']) / name
    required = [base / 'track.npz', base / 'cotracker.npz',
                Path(scene['sample_cache_dir']) / (name + '_S24_N128_seed125.npz'),
                Path(scene['triangulation_init_root']) / name / 'predictions/final_tracks.npz']
    if require_features:
        required += [Path(scene[k]) for k in PATH_KEYS[-3:]]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError('Required input files are missing:\n' + '\n'.join(missing))
    return scene


def sample_scene(scene):
    from datasets.exportdataset_3d import ExportDataset
    dataset = ExportDataset(dataset_location=scene['dataset_location'],
        specific_seq=scene['target_seq'], V=4, S=24, N=128, seed=125,
        sample_cache_dir=scene['sample_cache_dir'], require_sample_cache=True,
        expected_selected_views=scene['expected_selected_views'],
        triangulation_init_root=scene['triangulation_init_root'], cache_in_memory=False)
    return dataset[0]


def check_scene(scene):
    from track3d.models.encoders import FrozenFeatureCache
    sample = sample_scene(scene)
    rgbs = sample['rgbs'].permute(1, 0, 2, 3, 4).unsqueeze(0).contiguous()
    cache = FrozenFeatureCache(scene['semantic_cache_path'], 'dinov3')
    cache.get(rgbs)
    return {'scene': scene['target_seq'], 'rgb_shape': list(rgbs.shape),
            'point_slots': int(sample['sample_indices'].numel()),
            'semantic_cache_sha256': cache.sha256, 'optimizer_updates': 0}


def prepare_rgb(scene, output):
    import torch
    sample = sample_scene(scene)
    rgbs = sample['rgbs'].permute(1, 0, 2, 3, 4).unsqueeze(0).contiguous()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save({'schema': 'track3d_semantic_rgb_v1', 'rgbs': rgbs,
                    'provenance': {'scene': scene['target_seq'],
                                   'selected_views': scene['expected_selected_views']}}, stream)
    return {'rgb_bundle': str(path.resolve()), 'rgb_shape': list(rgbs.shape)}


def training_arguments(scene, output, device=0):
    args = json.loads((ROOT / 'configs/tracker.json').read_text())
    args.update(scene)
    args.update(ckpt_dir=str(output / 'initial'), log_dir=str(output / 'logs/initial'),
                run_name='run', device_ids=[device],
                convergence_config_path=str(ROOT / 'configs/convergence.json'))
    return args


def continuation_arguments(args, output, state):
    if state['reason'] == 'loss_plateau':
        return None
    if state['reason'] != 'budget_exhausted' or state['final_step'] != 240:
        raise ValueError('Unexpected optimization stopping state')
    return dict(args, max_iters=320, ckpt_dir=str(output / 'continuation'),
                log_dir=str(output / 'logs/continuation'),
                resume_checkpoint_path=str(output / 'initial/run/model-000000240.pth'))


def optimize(scene, output, device=0):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Scene optimization requires a CUDA GPU')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory: ' + str(output))
    preflight = check_scene(scene)
    output.mkdir(parents=True)
    (output / 'preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
    args = training_arguments(scene, output, device)
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONHASHSEED='0',
               OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    phases = []
    for phase in ('initial', 'continuation'):
        if args is None:
            break
        args_path = output / (phase + '_arguments.json')
        args_path.write_text(json.dumps(args, indent=2) + '\n')
        with (output / (phase + '.log')).open('x') as log:
            subprocess.run([sys.executable, '-u', '-m', 'track3d.cli.track',
                            '_fit', '--arguments', str(args_path)],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        folder = Path(args['ckpt_dir']) / 'run'
        state = json.loads((folder / 'convergence.json').read_text())
        phases.append({'phase': phase, 'convergence': state})
        if phase == 'initial':
            args = continuation_arguments(args, output, state)
    from track3d.utils.release_predictions import export_predictions
    prediction = folder / 'predictions/final_tracks.npz'
    arrays = export_predictions(prediction,
        Path(scene['dataset_location']) / scene['target_seq'] / 'track.npz', output / 'tracks.npz')
    result = {'prediction': str(prediction), 'tracks': str(output / 'tracks.npz'),
              'selected_step': state['selected_step'], 'actual_updates': state['final_step'],
              'stop_reason': state['reason'], 'selection': 'pseudo_supervised',
              'arrays': arrays, 'phases': phases}
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '_fit':
        internal = argparse.ArgumentParser()
        internal.add_argument('--arguments', required=True)
        args = internal.parse_args(sys.argv[2:])
        from train_optimize import main as fit
        fit(**json.loads(Path(args.arguments).read_text()))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for action in ('check', 'prepare-rgb', 'optimize'):
        p = commands.add_parser(action)
        p.add_argument('--scene-config', required=True)
        if action != 'check':
            p.add_argument('--output', required=True)
        if action == 'optimize':
            p.add_argument('--device', type=int, default=0, help='CUDA device index')
    p = commands.add_parser('export')
    p.add_argument('--prediction', required=True)
    p.add_argument('--cameras', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.action == 'export':
        from track3d.utils.release_predictions import export_predictions
        result = export_predictions(args.prediction, args.cameras, args.output)
    else:
        scene = load_scene(args.scene_config, require_features=args.action != 'prepare-rgb')
        if args.action == 'check':
            result = check_scene(scene)
        elif args.action == 'prepare-rgb':
            result = prepare_rgb(scene, args.output)
        else:
            result = optimize(scene, args.output, args.device)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
