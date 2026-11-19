"""Equation-level checks, including depth information lost by two-sided normalization."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from animimic.losses import frame_l2, joint_regularization, normalize_depth
from animimic.optimize import fit_joints, video_loss


class PaperObjectiveTests(unittest.TestCase):
    def test_norm_is_mean_of_per_frame_norms_and_has_finite_zero_gradient(self):
        pred = torch.tensor([[3., 4.], [0., 12.]], requires_grad=True)
        self.assertEqual(float(frame_l2(pred, torch.zeros_like(pred))), 8.5)
        zero = frame_l2(pred, pred.detach())
        zero.backward()
        torch.testing.assert_close(pred.grad, torch.zeros_like(pred))

    def test_regularizer_clips_nonroot_raw_parameters(self):
        rotation = torch.tensor([1., 0, 0, 0, 1, 0]).repeat(2, 2, 1)
        translation = torch.zeros(2, 2, 3)
        rotation[:, 0] = 100  # Root must not be penalized.
        translation[:, 0] = 100
        rotation[0, 1, 0] = 1.3
        translation[0, 1, 0] = .5
        # Frame 0: sqrt(.3^2 + .4^2) = .5; frame 1: zero.
        self.assertAlmostEqual(float(joint_regularization(rotation, translation,
            torch.tensor([True, False]), 1., .1)), .25, places=6)
        self.assertEqual(float(joint_regularization(rotation[:, :1], translation[:, :1],
            torch.tensor([True]))), 0.)

    def test_target_normalization_uses_fixed_camera_units(self):
        canonical = torch.tensor([4., 5., 6.], requires_grad=True)
        raw = torch.tensor([[10., 20., 30.], [100., 200., 300.]])
        actual = normalize_depth(raw, canonical, torch.ones(2, 3, dtype=torch.bool))
        torch.testing.assert_close(actual, canonical.detach()[None].expand(2, 3))
        self.assertFalse(actual.requires_grad)

    def test_rgb_includes_background_and_depth_retains_global_translation(self):
        class Renderer:
            size = (64, 64)
            def project(self, points):
                return points[..., :2]
            def depth(self, points):
                return points[..., 2]
            def __call__(self, vertices, faces, colors):
                return vertices.sum()*0+torch.zeros(1, 1, 3), vertices.sum()*0+torch.zeros(1, 1)
        vertices = torch.tensor([[2., 2., 4.], [3., 2., 5.], [2., 3., 6.]])
        rig = dict(vertices=vertices, faces=torch.tensor([[0, 1, 2]]), colors=torch.ones(3, 3))
        target = dict(triangles=torch.tensor([0, 0, 0]), barycentric=torch.eye(3),
            tracks=vertices[None, :, :2].repeat(2, 1, 1), depth=torch.tensor([[10., 20., 30.]]).repeat(2, 1),
            frames=torch.ones(2, 1, 1, 3), masks=torch.zeros(2, 1, 1),
            visibility=torch.ones(2, 3), confidence=torch.ones(2, 3))
        shift = torch.tensor(1., requires_grad=True)
        moved = vertices[None].repeat(2, 1, 1)+torch.stack((shift*0, shift*0, shift))
        losses = video_loss(moved, rig, target, Renderer())
        self.assertAlmostEqual(float(losses['rgb']), 3**.5, places=5)
        self.assertEqual(float(losses['track']), 0.)
        self.assertAlmostEqual(float(losses['depth']), 3**.5, places=5)
        losses['depth'].backward()
        self.assertAlmostEqual(float(shift.grad), 3**.5, places=5)
        # Confidence must not silently alter the Eq. 7 objective.
        target['confidence'].zero_()
        target['tracks'] = target['tracks']+1
        self.assertAlmostEqual(float(video_loss(moved.detach(), rig, target, Renderer())['track']), 6**.5, places=5)

    def test_all_joints_and_full_objective_are_optimized_from_first_step(self):
        local = torch.eye(4).repeat(2, 1, 1)
        rig = dict(vertices=torch.tensor([[1., 1., 0.]]), faces=torch.tensor([[0, 0, 0]]),
            colors=torch.ones(1, 3), weights=torch.tensor([[0., 1.]]),
            local=local, inverse_bind=local.clone(), parents=torch.tensor([-1, 0]))
        target = dict(frames=torch.zeros(2, 1, 1, 3))
        def objective(vertices, *args, **kwargs):
            return {'rgb': (vertices[..., 0]-2).square().mean()}
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()), \
             patch('animimic.optimize.video_loss', side_effect=objective) as objective_mock:
            output = Path(temporary)
            fit_joints(rig, target, None, output, dict(iterations=1))
            actual = np.load(output/'joint_trajectory.npz')
            self.assertGreater(actual['translation'][0, 1, 0], 0.)
            history = json.loads((output/'joint_losses.json').read_text())
            self.assertEqual(history[0]['phase'], 'full')
            self.assertEqual(objective_mock.call_args.args[3], None)
