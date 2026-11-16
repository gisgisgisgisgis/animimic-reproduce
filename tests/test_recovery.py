import unittest

import torch

from animimic.fem import matrix_free_step
from animimic.rig import fk, lbs, pose_local


class RecoveryTests(unittest.TestCase):
    def test_joint_motion_recovered_from_projection(self):
        rest = torch.eye(4).repeat(2, 1, 1)
        rest[1, 0, 3] = 1
        inverse = torch.linalg.inv(fk(rest, [-1, 0]))
        vertices = torch.tensor([[1., 1, 0], [1.5, 0, 0]])
        weights = torch.tensor([[0., 1], [0., 1]])
        rotation = torch.tensor([1., 0, 0, 0, 1, 0]).repeat(2, 1)
        truth = torch.zeros(2, 3)
        truth[1, :2] = torch.tensor([.2, .3])
        with torch.no_grad():
            target = lbs(vertices, weights, fk(pose_local(rest, rotation, truth),
                                               [-1, 0]), inverse)[:, :2]
        translation = torch.zeros(2, 3, requires_grad=True)
        opt = torch.optim.Adam([translation], lr=.03)
        initial = None
        for _ in range(150):
            opt.zero_grad()
            projected = lbs(vertices, weights, fk(pose_local(rest, rotation, translation),
                                                   [-1, 0]), inverse)[:, :2]
            loss = (projected-target).square().mean()
            if initial is None:
                initial = float(loss.detach())
            loss.backward()
            opt.step()
        self.assertLess(float(loss.detach()), initial*1e-4)

    def test_material_recovery_reduces_objective(self):
        rest = torch.tensor([[0., 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1]],
                            dtype=torch.float64)
        tets = torch.tensor([[0, 1, 2, 3]])
        mass = torch.ones(4, dtype=torch.float64)
        fixed = torch.tensor([True, True, True, False])
        target = rest.clone()
        target[1, 0] = 1.2
        def simulate(log_e):
            return matrix_free_step(rest, tets, mass, rest, torch.zeros_like(rest),
                                    log_e, .2, fixed, target)[0]
        truth = simulate(torch.tensor([4.], dtype=torch.float64)).detach()
        parameter = torch.tensor([2.], dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([parameter], lr=.15)
        initial = float((simulate(parameter)-truth).square().sum().detach())
        for _ in range(40):
            optimizer.zero_grad()
            loss = (simulate(parameter)-truth).square().sum()
            loss.backward()
            optimizer.step()
        self.assertLess(float(loss.detach()), initial*.1)
