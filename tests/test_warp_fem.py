import importlib.util
import unittest

import torch

from animimic.fem import BACKEND, elastic_energy, matrix_free_step, torch_elastic_energy


@unittest.skipUnless(importlib.util.find_spec("warp"), "Warp not installed")
class WarpFEMTests(unittest.TestCase):
    def test_energy_gradient_and_tangent_match_torch(self):
        rest = torch.tensor([[0., 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1]],
                            dtype=torch.float64)
        x = rest.clone()
        x[3, 2] = 1.1
        x.requires_grad_()
        tets = torch.tensor([[0, 1, 2, 3]])
        parameter = torch.tensor([3.], dtype=torch.float64, requires_grad=True)
        token = BACKEND.set("warp")
        try:
            actual = elastic_energy(x, rest, tets, parameter)
            expected = torch_elastic_energy(x, rest, tets, parameter)
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
            ga = torch.autograd.grad(actual, x, create_graph=True)[0]
            ge = torch.autograd.grad(expected, x, create_graph=True)[0]
            torch.testing.assert_close(ga, ge)
            direction = torch.ones_like(x)
            ha = torch.autograd.grad(ga, x, direction)[0]
            he = torch.autograd.grad(ge, x, direction)[0]
            torch.testing.assert_close(ha, he)
            fixed = torch.tensor([True, True, True, False])
            target = rest.clone()
            target[1, 0] = 1.1
            result, _ = matrix_free_step(rest, tets, torch.ones(4, dtype=torch.float64),
                rest, torch.zeros_like(rest), parameter, .1, fixed, target)
            gradient = torch.autograd.grad(result[3, 2], parameter)[0]
            self.assertTrue(torch.isfinite(gradient).all())
        finally:
            BACKEND.reset(token)
