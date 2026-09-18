"""CPU integration checks for portable input and output contracts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from track3d.cli.track import load_scene, training_arguments, continuation_arguments, optimize
from track3d.utils.release_predictions import export_predictions


class PublicInterfaceTest(unittest.TestCase):
    def scene_config(self, root):
        scene = dict(dataset_location='data', target_seq='scene',
            sample_cache_dir='points', expected_selected_views=[1, 7, 14, 20],
            triangulation_init_root='triangulation', semantic_cache_path='features.pt',
            common_init_checkpoint='common.pth', cotracker_encoder_checkpoint='encoder.pth')
        for name in ('data/scene/track.npz', 'data/scene/cotracker.npz',
                     'points/scene_S24_N128_seed125.npz',
                     'triangulation/scene/predictions/final_tracks.npz'):
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
        path = root / 'scene.json'
        path.write_text(json.dumps(scene))
        return path, scene

    def test_rgb_preparation_allows_features_to_be_built_later(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = self.scene_config(root)
            scene = load_scene(path, require_features=False)
            self.assertEqual(scene['dataset_location'], str(root / 'data'))
            with self.assertRaises(FileNotFoundError):
                load_scene(path)

    def test_wrong_point_identity_cache_is_not_silently_regenerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = self.scene_config(root)
            (root / 'points/scene_S24_N128_seed125.npz').unlink()
            (root / 'points/scene_S24_N128_seed0.npz').touch()
            with self.assertRaises(FileNotFoundError):
                load_scene(path, require_features=False)

    def test_unknown_settings_and_duplicate_camera_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, scene = self.scene_config(root)
            for invalid in (dict(scene, learning_rate=1),
                            dict(scene, expected_selected_views=[0, 0, 1, 2])):
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    load_scene(path, require_features=False)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'keep.txt'
            marker.write_text('existing result')
            with patch('torch.cuda.is_available', return_value=True):
                with self.assertRaises(FileExistsError):
                    optimize({}, tmp)
            self.assertEqual(marker.read_text(), 'existing result')

    def test_budget_continuation_uses_terminal_state_and_preserves_inputs(self):
        output = Path('/tmp/example-run')
        args = training_arguments({'semantic_cache_path': '/tmp/features.pt'}, output)
        state = dict(reason='budget_exhausted', final_step=240, selected_step=70)
        continued = continuation_arguments(args, output, state)
        self.assertEqual(args['max_iters'], 240)
        self.assertEqual(continued['max_iters'], 320)
        self.assertTrue(continued['resume_checkpoint_path'].endswith('model-000000240.pth'))
        self.assertEqual(continued['semantic_cache_path'], args['semantic_cache_path'])
        self.assertEqual(continued['fnet_warmup_steps'], 0)
        self.assertIsNone(continuation_arguments(args, output,
            dict(reason='loss_plateau', final_step=160)))
        with self.assertRaises(ValueError):
            continuation_arguments(args, output, dict(reason='budget_exhausted', final_step=100))

    def test_export_requires_only_camera_parameters_and_preserves_query_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xyz = np.array([[[[2., 4., 2.], [1., 1., -1.]],
                             [[4., 4., 2.], [1., 1., -1.]]]], dtype=np.float32)
            np.savez(root / 'prediction.npz', trajs_3d=xyz,
                visibility=np.ones((2, 2), bool),
                query_points=np.array([[[0, 2, 4, 2], [1, 1, 1, -1]]]),
                sample_indices=np.array([[7, 9]]), step=20, visibility_source='triangulation')
            k = np.tile(np.eye(3), (2, 2, 1, 1))
            rt = np.tile(np.eye(4), (2, 2, 1, 1))
            rt[:, 1, 0, 3] = 2
            np.savez(root / 'cameras.npz', cam_k=k, cam_rt=rt, selected_views=[1, 7])
            report = export_predictions(root / 'prediction.npz', root / 'cameras.npz', root / 'out.npz')
            self.assertFalse(report['gt_used'])
            with np.load(root / 'out.npz') as z:
                np.testing.assert_array_equal(z['sample_indices'], [7, 9])
                np.testing.assert_array_equal(z['selected_views'], [1, 7])
                np.testing.assert_array_equal(z['tracks_2d'][0, :, 0], [[1, 2], [2, 2]])
                self.assertTrue(np.isnan(z['tracks_2d'][:, :, 1]).all())
                self.assertFalse(z['query_active'][0, 1])
                self.assertTrue(z['query_active'][1, 1])
            with self.assertRaises(FileExistsError):
                export_predictions(root / 'prediction.npz', root / 'cameras.npz', root / 'out.npz')


if __name__ == '__main__':
    unittest.main()
