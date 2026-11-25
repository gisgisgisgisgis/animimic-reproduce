"""Verify the pinned optimizer shares the exact scene, observations and frame schedule."""
import contextlib
import io
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from animimic.compare_optimizers import artifact, load_reference, run_reference, SeededTorch
from animimic.core import file_hash
from animimic.optimize import fit_joints, objective_weights
from animimic.reference_stage1 import SOURCE_HASHES


class Renderer:
    size = (64, 96)

    def __init__(self):
        self.frames = []

    def project(self, points):
        # Match the real renderer's single-frame API.
        if points.ndim != 2:
            raise ValueError('project expects N,3')
        return points[..., :2]*16+32

    def depth(self, points):
        return points[..., 2]

    def __call__(self, vertices, faces, colors):
        self.frames.append(vertices.detach().clone())
        return vertices.mean(0).sigmoid()[None, None], vertices.mean().sigmoid()[None, None]


def fixture():
    n = 5
    vertices = torch.tensor([[0., 0., 4.], [.1, 0., 5.], [0., .1, 6.]])
    rig = dict(vertices=vertices, faces=torch.tensor([[0, 1, 2]]), colors=torch.ones(3, 3),
               weights=torch.tensor([[0., 1.]]).repeat(3, 1), local=torch.eye(4).repeat(2, 1, 1),
               inverse_bind=torch.eye(4).repeat(2, 1, 1), parents=torch.tensor([-1, 0]))
    target = dict(frames=torch.ones(n, 1, 1, 3), masks=torch.ones(n, 1, 1),
                  triangles=torch.tensor([0, 0, 0]), barycentric=torch.eye(3),
                  tracks=Renderer().project(vertices)[None].repeat(n, 1, 1),
                  depth=vertices[None, :, 2].repeat(n, 1), confidence=torch.ones(n, 3),
                  frame_times=torch.linspace(0, 1, n))
    target['tracks'][:, :, 0] += torch.arange(n)[:, None]*.4
    return rig, target


class OptimizerComparisonTests(unittest.TestCase):
    def test_reference_rng_uses_same_permutations(self):
        actual = SeededTorch(7)
        expected = torch.Generator().manual_seed(7)
        for _ in range(5):
            torch.testing.assert_close(actual.randperm(25, device='cpu'),
                                       torch.randperm(25, generator=expected))

    def test_reference_and_adaptation_match_on_shared_scene(self):
        rig, target = fixture()
        config = dict(objective='normalized', iterations=8, batch_frames=2, seed=4,
                      rotation_bound=1., translation_bound=.01)
        config['loss_weights'] = objective_weights(config)
        config['loss_weights']['rigid'] = 10.
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            source, provenance = load_reference(root/'source')
            self.assertEqual(set(provenance['source_hashes']), {'rig', 'losses', 'stage1'})
            adapted, reference = root/'adapted', root/'reference'
            adapted.mkdir()
            reference.mkdir()
            first, second = Renderer(), Renderer()
            fit_joints(rig, target, first, adapted, config)
            run_reference(source, rig, target, second, reference, config)
            self.assertEqual(len(first.frames), config['iterations']*config['batch_frames'])
            self.assertEqual(len(first.frames), len(second.frames))
            for a, b in zip(first.frames, second.frames):
                torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
            with np.load(adapted/'joint_trajectory.npz') as a, np.load(reference/'joint_trajectory.npz') as b:
                np.testing.assert_allclose(a['vertices'], b['vertices'], atol=2e-6, rtol=2e-6)
            self.assertTrue(json.loads((reference/'joint_losses.json').read_text()))

    def test_reference_loads_without_git_or_network(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch('subprocess.Popen', side_effect=AssertionError('No subprocess permitted')):
            source, provenance = load_reference(Path(temporary)/'source')
            self.assertEqual(provenance['source_hashes'], SOURCE_HASHES)
            self.assertTrue(callable(source.run_stage1))
            cached = Path(source.__file__).with_name('rig.py')
            cached.write_text('corrupted cache')
            load_reference(Path(temporary)/'source')
            self.assertEqual(file_hash(cached), SOURCE_HASHES['rig'])

    @unittest.skipUnless(importlib.util.find_spec('pytorch3d'), 'PyTorch3D unavailable')
    def test_cli_runs_from_a_directory_without_git(self):
        import imageio.v2 as iio
        from animimic.core import save_json
        from animimic.render import Renderer as MeshRenderer

        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='animimic no git ') as temporary:
            root = Path(temporary)
            shutil.copytree(repository/'src/animimic', root/'src/animimic',
                            ignore=shutil.ignore_patterns('__pycache__'))
            self.assertFalse((root/'.git').exists())
            vertices = np.array([[-.2, -.2, 0.], [.2, -.2, 0.], [0., .2, .1]], dtype=np.float32)
            camera = dict(width=32, height=32)
            renderer = MeshRenderer('cpu', **camera)
            xy = renderer.project(torch.tensor(vertices)).numpy()
            manifest = {'config': {'camera': camera, 'joints': dict(objective='normalized',
                        iterations=1, batch_frames=1)}, 'stages': {}}
            prepared = root/'.cache/animimic/prepare/key'
            supervised = root/'.cache/animimic/supervision/key'
            prepared.mkdir(parents=True)
            supervised.mkdir(parents=True)
            np.savez(prepared/'rig.npz', vertices=vertices, faces=np.array([[0, 1, 2]]),
                     colors=np.ones((3, 3), dtype=np.float32),
                     weights=np.tile([0., 1.], (3, 1)), local=np.tile(np.eye(4), (2, 1, 1)),
                     inverse_bind=np.tile(np.eye(4), (2, 1, 1)), parents=np.array([-1, 0]))
            np.savez(supervised/'supervision.npz', frames=np.ones((3, 32, 32, 3)),
                     masks=np.ones((3, 32, 32)), triangles=np.array([0, 0, 0]), barycentric=np.eye(3),
                     tracks=np.tile(xy, (3, 1, 1)), depth=np.tile([1., 2., 3.], (3, 1)),
                     visibility=np.ones((3, 3)), confidence=np.ones((3, 3)), indices=np.arange(3))
            for stage, folder, name in [('prepare', prepared, 'rig.npz'),
                                        ('supervision', supervised, 'supervision.npz')]:
                manifest['stages'][stage] = {'directory': str(folder.relative_to(root)),
                                             'artifacts': {name: file_hash(folder/name)}}
            run = root/'runs/fixture'
            run.mkdir(parents=True)
            with iio.get_writer(str(run/'reference.mp4'), fps=8) as writer:
                for _ in range(3):
                    writer.append_data(np.full((32, 32, 3), 255, dtype=np.uint8))
            manifest['stages']['video'] = {'artifacts': {'reference.mp4': file_hash(run/'reference.mp4')}}
            save_json(run/'manifest.json', manifest)
            result = subprocess.run([sys.executable, '-m', 'animimic.compare_optimizers',
                                     str(run), '--device', 'cpu', '--no-previews'], cwd=root,
                                    env={**os.environ, 'PYTHONPATH': str(root/'src'),
                                         'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'},
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            reports = list((run/'optimizer-comparison').glob('*/comparison.json'))
            self.assertEqual(len(reports), 1)
            self.assertEqual(set(json.loads(reports[0].read_text())['arms']), {'adapted', 'reference'})

    def test_transferred_cache_requires_original_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root/'.cache/animimic/animimic/supervision/key'
            nested.mkdir(parents=True)
            (nested/'supervision.npz').write_bytes(b'original')
            manifest = {'stages': {'supervision': {'directory': '.cache/animimic/supervision/key',
                        'artifacts': {'supervision.npz': file_hash(nested/'supervision.npz')}}}}
            self.assertEqual(artifact(manifest, 'supervision', 'supervision.npz', root), nested/'supervision.npz')
            (nested/'supervision.npz').write_bytes(b'wrong run')
            with self.assertRaises(FileNotFoundError):
                artifact(manifest, 'supervision', 'supervision.npz', root)
