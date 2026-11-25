"""Native source integrity, camera conversion and unmodified Stage 1 execution."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT/'native-stage1'
if not (NATIVE/'run_existing_video.py').is_file():
    raise unittest.SkipTest('The separate native-stage1 runner is unavailable')
spec = importlib.util.spec_from_file_location('native_existing_video', NATIVE/'run_existing_video.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class NativeInputTests(unittest.TestCase):
    def test_pinned_files_and_similarity(self):
        self.assertEqual(len(entry.verify_source()['files']), 19)
        rng = np.random.default_rng(4)
        vertices = rng.normal(size=(30, 3))
        angle = .4
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        target = .76*vertices@rotation+np.array([.1, -.2, .3])
        scale, actual, offset, error = entry.similarity(vertices, target)
        np.testing.assert_allclose(actual, rotation, atol=1e-10)
        self.assertAlmostEqual(scale, .76)
        self.assertLess(error, 1e-10)
        with self.assertRaises(ValueError):
            entry.similarity(vertices, target*np.array([1., 2., 1.]))

    def test_runner_selects_original_scripts_without_stage2(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root/'run'
            run.mkdir()
            (run/'manifest.json').write_text(json.dumps({'config': {'asset': 'fixture'}}))
            output = root/'native'
            seq = output/'sequence'
            seq.mkdir(parents=True)
            for name in ('camera.json', 'rigged.glb', 'tracks.npz', 'depth.npz'):
                (seq/name).touch()
            tracker = root/'tracker/nets'
            tracker.mkdir(parents=True)
            (tracker/'alltracker.py').touch()
            args = ['test', str(run), '--output', str(output), '--device', 'cpu',
                    '--alltracker-repo', str(tracker.parent)]
            with patch.object(sys, 'argv', args), patch.object(entry, 'prepare', return_value={'duration': 5.}), \
                 patch.object(entry.subprocess, 'run') as invoke:
                entry.main()
            commands = [c.args[0] for c in invoke.call_args_list]
            self.assertEqual([Path(c[1]).name for c in commands], [
                'prep_01_render_view.py', 'prep_02_import_video.py', 'prep_04_sam2.py',
                'prep_05_alltracker.py', 'prep_06_vggt.py', 'run_animamimic.py', 'export_animated_glb.py',
                'export_preview.py'])
            stage1 = commands[-3]
            self.assertNotIn('--stage2', stage1)
            self.assertEqual(stage1[stage1.index('--backend')+1], 'raster')
            self.assertEqual(stage1[stage1.index('--iters')+1], '3000')
            self.assertEqual(stage1[stage1.index('--batch_frames')+1], '4')


@unittest.skipUnless(importlib.util.find_spec('pygltflib'), 'Native GLB dependency unavailable')
class NativeExecutionTests(unittest.TestCase):
    def test_preview_interpolates_rotations_without_shortening_bones(self):
        import torch
        from scipy.spatial.transform import Rotation
        with patch.object(sys, 'path', [str(NATIVE), *sys.path]):
            from export_preview import interpolate_motion
            from animamimic.rig import matrix_to_rotation_6d, rotation_6d_to_matrix
            matrices = Rotation.from_euler('z', np.array([[0.], [90.], [180.]]), degrees=True).as_matrix().astype(np.float32)
            r6d = matrix_to_rotation_6d(torch.from_numpy(matrices[:, None])).numpy()
            translations = np.zeros((3, 1, 3), dtype=np.float32)
            translations[:, 0, 0] = [0., 2., 4.]
            rotation, offset, indices = interpolate_motion(r6d, translations, 5)
            actual = rotation_6d_to_matrix(torch.from_numpy(rotation)).numpy()
            np.testing.assert_array_equal(indices, [0, 2, 4])
            np.testing.assert_allclose(actual[indices, 0], matrices, atol=1e-6)
            np.testing.assert_allclose(np.linalg.det(actual), 1., atol=1e-6)
            np.testing.assert_allclose(actual[1, 0]@[1., 0., 0.], [2**-.5, 2**-.5, 0.], atol=1e-6)
            np.testing.assert_allclose(offset[:, 0, 0], np.arange(5))

    def test_native_fk_renderer_optimizer_and_glb_export(self):
        import imageio.v2 as iio
        import torch
        from PIL import Image
        from animimic.export import export_glb
        with tempfile.TemporaryDirectory() as temporary, patch.object(sys, 'path', [str(NATIVE), *sys.path]):
            from animamimic.camera import Camera
            from animamimic.gltf_rig import RiggedGLB
            from animamimic.render import RasterRenderer
            from animamimic.rig import identity_pose, forward_kinematics, linear_blend_skinning
            root = Path(temporary)
            seq, output = root/'sequence', root/'output'
            seq.mkdir()
            vertices = np.array([[-.2, -.2, 0.], [.2, -.2, 0.], [0., .2, .1]])
            local = np.tile(np.eye(4), (2, 1, 1))
            rig = dict(vertices=vertices, faces=np.array([[0, 1, 2]]), colors=np.ones((3, 3))*.7,
                       weights=np.tile([0., 1.], (3, 1)), inverse_bind=local.copy(), parents=np.array([-1, 0]))
            export_glb(seq/'rigged.glb', rig, local[None].repeat(3, 0), 12)
            model = RiggedGLB(seq/'rigged.glb')
            mesh = model.mesh()
            camera = Camera.look_at(2.2, 10., 0., 35., 32)
            camera.to_json(seq/'camera.json')
            native_rig = model.rig()
            rotation, translation = identity_pose(3, native_rig.num_joints, 'cpu')
            a, b = forward_kinematics(native_rig, rotation, translation)
            static = linear_blend_skinning(native_rig, mesh.verts, a, b)
            torch.testing.assert_close(static, mesh.verts[None].expand(3, -1, -1))
            renderer = RasterRenderer(mesh, camera, torch.device('cpu'), sigma=1e-5)
            with torch.no_grad():
                rgb, mask = renderer(mesh.verts[None])
            ys, xs = torch.where(mask[0] > .9)
            self.assertGreater(len(xs), 6)
            pick = torch.linspace(0, len(xs)-1, 6).long()
            xy = torch.stack([xs[pick], ys[pick]], -1).float().numpy()+.5
            for folder, values in [('frames', (rgb[0].clamp(0, 1).numpy()*255).astype(np.uint8)),
                                   ('masks', (mask[0].numpy()>.5).astype(np.uint8)*255)]:
                (seq/folder).mkdir()
                for i in range(3):
                    Image.fromarray(values).save(seq/folder/f'{i:05d}.png')
            np.savez(seq/'tracks.npz', xy=np.tile(xy, (3, 1, 1)), visible=np.ones((3, 6), bool))
            depth = 1+np.arange(32)[None, :]*.01+np.arange(32)[:, None]*.02
            np.savez(seq/'depth.npz', depth=np.tile(depth, (3, 1, 1)).astype(np.float32))
            (seq/'input-manifest.json').write_text(json.dumps({'source_frames': 5, 'fps': 8.}))
            with iio.get_writer(str(seq/'reference-square.mkv'), fps=8, codec='ffv1', pixelformat='bgr0') as writer:
                for _ in range(5):
                    writer.append_data((rgb[0].clamp(0, 1).numpy()*255).astype(np.uint8))
            env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
            commands = [[sys.executable, str(NATIVE/'scripts/run_animamimic.py'), '--seq', str(seq),
                         '--out', str(output), '--device', 'cpu', '--backend', 'raster', '--image_size', '32',
                         '--iters', '1', '--batch_frames', '1', '--sigma', '1e-5', '--depth_norm', 'per_frame'],
                        [sys.executable, str(NATIVE/'scripts/export_animated_glb.py'), '--seq', str(seq),
                         '--stage1', str(output/'stage1.npz'), '--duration', '.25', '--out', str(output/'joint.glb')],
                        [sys.executable, str(NATIVE/'export_preview.py'), '--seq', str(seq),
                         '--result', str(output/'stage1.npz'), '--device', 'cpu', '--image-size', '32']]
            for command in commands:
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertTrue((output/'stage1_vs_video.mp4').is_file())
            self.assertTrue((output/'joint.glb').is_file())
            with iio.get_reader(str(output/'stage1_vs_video_native_fps.mp4')) as reader:
                self.assertEqual(reader.count_frames(), 5)
                self.assertEqual(reader.get_meta_data()['fps'], 8)
            with np.load(output/'stage1.npz') as result:
                self.assertTrue(np.isfinite(result['verts']).all())
