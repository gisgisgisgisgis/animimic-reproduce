import unittest

import torch

from animimic.fem import elastic_energy, matrix_free_step, volumes


class FEMTests(unittest.TestCase):
    def setUp(self):
        self.rest = torch.tensor([[0., 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1]],
                                 dtype=torch.float64)
        self.tets = torch.tensor([[0, 1, 2, 3]])
        self.mass = torch.ones(4, dtype=torch.float64)
        self.fixed = torch.tensor([True, True, True, False])

    def test_identity_has_zero_force_and_finite_tangent(self):
        x = self.rest.clone().requires_grad_()
        e = torch.tensor([3.], dtype=torch.float64)
        energy = elastic_energy(x, self.rest, self.tets, e)
        grad = torch.autograd.grad(energy, x, create_graph=True)[0]
        torch.testing.assert_close(grad, torch.zeros_like(grad))
        hessian = torch.autograd.functional.hessian(
            lambda p: elastic_energy(p, self.rest, self.tets, e), x)
        self.assertTrue(torch.isfinite(hessian).all())

    def simulate(self, e):
        target = self.rest.clone()
        target[1, 0] = 1.1
        return matrix_free_step(self.rest, self.tets, self.mass, self.rest,
            torch.zeros_like(self.rest), e, 0.1, self.fixed, target)[0]

    def test_material_gradient_matches_finite_difference(self):
        e = torch.tensor([3.], dtype=torch.float64, requires_grad=True)
        x = self.simulate(e)
        self.assertGreater(float(volumes(x, self.tets).detach().min()), 0)
        loss = x[3, 2]
        analytic = torch.autograd.grad(loss, e)[0]
        epsilon = 1e-3
        numerical = (self.simulate(e.detach()+epsilon)[3, 2]
                     - self.simulate(e.detach()-epsilon)[3, 2])/(2*epsilon)
        torch.testing.assert_close(analytic[0], numerical, atol=1e-5, rtol=1e-2)

    def test_rest_state_stays_at_rest(self):
        x, v = matrix_free_step(self.rest, self.tets, self.mass, self.rest,
            torch.zeros_like(self.rest), torch.tensor([3.], dtype=torch.float64),
            .1, self.fixed, self.rest)
        torch.testing.assert_close(x, self.rest)
        torch.testing.assert_close(v, torch.zeros_like(v))

    def test_multistep_adjoint_matches_finite_difference(self):
        def simulate(e):
            target = self.rest.clone()
            target[1, 0] = 1.1
            x, velocity = self.rest, torch.zeros_like(self.rest)
            for _ in range(2):
                x, velocity = matrix_free_step(self.rest, self.tets, self.mass,
                    x, velocity, e, .1, self.fixed, target)
            return x[3, 2]
        e = torch.tensor([3.], dtype=torch.float64, requires_grad=True)
        gradient = torch.autograd.grad(simulate(e), e)[0]
        numerical = (simulate(e.detach()+1e-3)-simulate(e.detach()-1e-3))/2e-3
        torch.testing.assert_close(gradient[0], numerical, rtol=1e-2, atol=1e-5)
