import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from animimic.losses import stationary_track_weights, temporal_acceleration
from animimic.optimize import fit_joints
from animimic.rig import fk, lbs, pose_local


class JointStabilityTests(unittest.TestCase):
    def test_acceleration_is_consistent_across_sample_rates(self):
        for count in (25, 49, 121):
            times = torch.linspace(0, 5, count, dtype=torch.float64)
            constant_velocity = times[:, None]*.3
            torch.testing.assert_close(temporal_acceleration(constant_velocity, times),
                                       torch.tensor(0., dtype=torch.float64), atol=1e-20, rtol=0)
            # x(t) = t^2 has acceleration 2, hence mean squared acceleration 4.
            torch.testing.assert_close(temporal_acceleration(times[:, None]**2, times),
                                       torch.tensor(4., dtype=torch.float64))

    def test_stationary_prior_excludes_moving_and_unreliable_tracks(self):
        tracks = torch.zeros(5, 3, 2)
        tracks[:, 1, 0] = torch.arange(5)*3
        weight = torch.ones(5, 3)
        weight[2:, 2] = .1
        actual = stationary_track_weights(tracks, weight)
        torch.testing.assert_close(actual[:, 0], torch.ones(5))
        torch.testing.assert_close(actual[:, 1:], torch.zeros(5, 2))

    def test_noisy_depth_does_not_stretch_bones_or_shake_stationary_foot(self):
        torch.manual_seed(4)
        count = 25
        times = torch.linspace(0, 5, count)
        local = torch.eye(4).repeat(3, 1, 1)
        local[1, :3, 3] = torch.tensor([-.2, -.2, 0.])
        local[2, :3, 3] = torch.tensor([.2, .2, 0.])
        vertices = torch.tensor([[-.22, -.35, 0], [-.18, -.35, .01], [-.2, -.4, -.01],
                                 [.2, .4, 0], [.25, .4, 0], [.25, .45, 0]])
        parents = torch.tensor([-1, 0, 0])
        weights = torch.zeros(6, 3)
        weights[:3, 1], weights[3:, 2] = 1, 1
        rig = dict(vertices=vertices, faces=torch.tensor([[0, 1, 2], [3, 4, 5]]),
                   weights=weights, colors=torch.ones_like(vertices), local=local,
                   inverse_bind=torch.linalg.inv(fk(local, parents.tolist())), parents=parents)
        rotation = torch.tensor([1., 0, 0, 0, 1, 0]).repeat(count, 3, 1)
        angle = .4*torch.sin(times/5*torch.pi)
        rotation[:, 2, 0], rotation[:, 2, 1] = angle.cos(), angle.sin()
        rotation[:, 2, 3], rotation[:, 2, 4] = -angle.sin(), angle.cos()
        truth = lbs(vertices, weights, fk(pose_local(local, rotation, torch.zeros(count, 3, 3)),
                                         parents.tolist()), rig["inverse_bind"])
        noisy = truth.clone()
        noisy[:, :3, 2] += torch.randn(count, 1)*.025
        noisy[:, :3, 0] += torch.randn(count, 1)*.002
        target = dict(frames=torch.zeros(count, 1, 1, 3), tracks=noisy[:, :, :2]*100,
                      visibility=torch.ones(count, 6), confidence=torch.ones(count, 6),
                      triangles=torch.tensor([0, 0, 0, 1, 1, 1]),
                      barycentric=torch.eye(3).repeat(2, 1), frame_times=times)
        def objective(predicted, *args):
            return {"track": (predicted[..., :2]-noisy[..., :2]).square().mean(),
                    "depth": (predicted[..., 2]-noisy[..., 2]).square().mean()}
        settings = dict(root_iterations=0, geometry_iterations=160, full_iterations=0,
                        learning_rate=.015)
        def solve(legacy):
            loss_weights = dict(track=1000., depth=1000.)
            if legacy:
                loss_weights.update(rotation_smooth=0., root_smooth=0., stationary=0.)
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                 patch("animimic.optimize.video_loss", side_effect=objective):
                fit_joints(rig, target, None, Path(directory),
                           {**settings, "optimize_joint_translations": legacy, "loss_weights": loss_weights})
                return dict(np.load(Path(directory)/"joint_trajectory.npz"))
        before, after = solve(True), solve(False)
        jitter_before = np.mean(np.diff(before["vertices"][:, :3], n=2, axis=0)**2)
        jitter_after = np.mean(np.diff(after["vertices"][:, :3], n=2, axis=0)**2)
        self.assertLess(jitter_after, jitter_before*.4)
        np.testing.assert_allclose(after["local"][:, 1:, :3, 3],
                                   np.broadcast_to(local[None, 1:, :3, 3], (count, 2, 3)), atol=1e-7)
        head_error = np.linalg.norm(after["vertices"][:, 3:]-truth[:, 3:].numpy(), axis=-1).mean()
        self.assertLess(head_error, .015)
        self.assertTrue(np.isfinite(after["vertices"]).all())
