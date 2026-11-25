"""Normalized objective, masking, geometry preservation and fitting smoke tests."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from animimic.core import load_config
from animimic.evaluate import deformation_metrics
from animimic.losses import edge_rigidity, standardize_depth, offset_regularization, mesh_edges
from animimic.optimize import fit_joints, objective_weights, video_loss


class Renderer:
    size = (64, 96)

    def __init__(self):
        self.calls = 0

    def project(self, points):
        return points[..., :2]*16+32

    def depth(self, points):
        return points[..., 2]

    def __call__(self, vertices, faces, colors):
        self.calls += 1
        zero = vertices.sum()*0
        return zero+torch.ones(1, 1, 3), zero+torch.ones(1, 1)


def fixture(count=5):
    vertices = torch.tensor([[0., 0., 4.], [.1, 0., 5.], [0., .1, 6.]])
    rig = dict(vertices=vertices, faces=torch.tensor([[0, 1, 2]]), colors=torch.ones(3, 3),
               weights=torch.ones(3, 1), local=torch.eye(4)[None],
               inverse_bind=torch.eye(4)[None], parents=torch.tensor([-1]))
    target = dict(frames=torch.ones(count, 1, 1, 3), masks=torch.ones(count, 1, 1),
                  triangles=torch.tensor([0, 0, 0]), barycentric=torch.eye(3),
                  tracks=Renderer().project(vertices)[None].repeat(count, 1, 1),
                  depth=vertices[None, :, 2].repeat(count, 1),
                  visibility=torch.ones(count, 3), confidence=torch.ones(count, 3),
                  frame_times=torch.linspace(0, 1, count))
    return rig, target


class NormalizedObjectiveTests(unittest.TestCase):
    def test_depth_alignment_handles_missing_and_single_tracks(self):
        raw = torch.tensor([[10., 20., 30.], [100., 200., 300.], [9., 8., 7.], [1., 2., 3.]])
        valid = torch.tensor([[True]*3, [True]*3, [True, False, False], [False]*3])
        actual = standardize_depth(raw, torch.tensor([4., 5., 6.]), valid)
        torch.testing.assert_close(actual[:2], torch.tensor([[4., 5., 6.]]).repeat(2, 1))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(float(actual[2, 0]), 5.)

    def test_regularization_uses_identity_offsets_and_excludes_root(self):
        rotation = torch.tensor([1., 0, 0, 0, 1, 0]).repeat(2, 2, 1)
        translation = torch.zeros(2, 2, 3)
        rotation[:, 0], translation[:, 0] = 100., 100.
        rotation[0, 1, 0] += 1.3
        translation[0, 1, 0] = .05
        actual = offset_regularization(rotation, translation, torch.tensor([True, False]))
        self.assertAlmostEqual(float(actual), (.3**2+.04**2)/2, places=6)

    def test_rigidity_preserves_rigid_motion_and_penalizes_stretch(self):
        rest = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., .001, 0.]])
        edges, lengths = mesh_edges(rest, torch.tensor([[0, 1, 2]]))
        self.assertEqual(float(edge_rigidity(rest[None], edges, lengths)), 0.)
        angle = torch.tensor(.7)
        rotation = torch.tensor([[angle.cos(), -angle.sin(), 0.],
                                 [angle.sin(), angle.cos(), 0.], [0., 0., 1.]])
        moved = (rest@rotation.T+2)[None]
        self.assertLess(float(edge_rigidity(moved, edges, lengths)), 1e-10)
        stretched = (rest*1.2)[None].requires_grad_()
        loss = edge_rigidity(stretched, edges, lengths)
        self.assertGreater(float(loss), .01)
        loss.backward()
        self.assertTrue(torch.isfinite(stretched.grad).all())

    def test_masking_rectangular_scale_and_render_minibatch(self):
        rig, target = fixture()
        predicted = rig['vertices'][None].repeat(5, 1, 1).requires_grad_()
        target['tracks'][1:, 1:] += 20
        target['confidence'][1:, 1:] = 0
        target['tracks'][1:, 0, 0] += 3
        renderer = Renderer()
        losses = video_loss(predicted, rig, target, renderer, config=dict(objective='normalized'),
                            render_indices=torch.tensor([0, 2]))
        self.assertEqual(renderer.calls, 2)
        self.assertAlmostEqual(float(losses['track']), 12/(7*96), places=6)
        sum(losses.values()).backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        # No usable observations after frame 0 must stay finite, including backward.
        target['confidence'][1:] = 0
        losses = video_loss(predicted, rig, target, renderer, config=dict(objective='normalized'))
        self.assertEqual(float(losses['track']), 0.)
        self.assertTrue(all(torch.isfinite(v) for v in losses.values()))

    def test_seeded_fit_writes_finite_motion_and_diagnostics(self):
        rig, target = fixture()
        target['tracks'][1:, :, 0] += 1
        config = dict(objective='normalized', iterations=3, batch_frames=2, seed=4,
                      loss_weights=dict(rigid=10.))
        results = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                renderer = Renderer()
                files = fit_joints(rig, target, renderer, Path(directory), config)
                self.assertEqual(renderer.calls, 6)
                self.assertTrue(all((Path(directory)/f).is_file() for f in files))
                result = dict(np.load(Path(directory)/'joint_trajectory.npz'))
                history = json.loads((Path(directory)/'joint_losses.json').read_text())
                self.assertIn('rigid', history[-1])
                self.assertGreater(np.abs(result['translation']).max(), 0.)
                self.assertTrue(np.isfinite(result['vertices']).all())
                diagnostics = json.loads((Path(directory)/'joint_diagnostics.json').read_text())
                self.assertTrue(all(np.isfinite(v) for v in diagnostics.values()))
                results.append(result['vertices'])
        np.testing.assert_array_equal(*results)

    def test_grouped_tracks_preset_uses_normalized_objective(self):
        root = Path(__file__).resolve().parents[1]/'configs/experiments'
        config = load_config(root/'horse-textured-wan-leftforeleg-grouped-tracks.yaml')
        self.assertEqual(config['joints']['objective'], 'normalized')
        self.assertGreater(objective_weights(config['joints'])['rigid'], 0)

    def test_unknown_objective_is_rejected(self):
        with self.assertRaises(ValueError):
            objective_weights(dict(objective='typo'))

    def test_diagnostics_zero_for_static_mesh(self):
        rig, target = fixture()
        result = deformation_metrics(rig['vertices'][None].repeat(5, 1, 1), rig, target['frame_times'])
        self.assertTrue(all(v == 0 for v in result.values()))
