import unittest

import torch

from animimic.fem import (elastic_energy, matrix_free_step, volumes,
                         initial_state, adaptive_step, FEMConvergenceError)


class FEMTests(unittest.TestCase):
    def test_analytic_element_hessian_matches_energy_derivative(self):
        from animimic.fem import element_hessians
        for scale in (1., .85, 1.2):
            x = self.rest.clone()*scale
            x[1, 1] += .1
            e = torch.tensor([3.], dtype=torch.float64)
            actual = element_hessians(x, self.rest, self.tets, e)[0]
            expected = torch.autograd.functional.hessian(
                lambda p: elastic_energy(p, self.rest, self.tets, e), x).reshape(12, 12)
            torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)

    def test_warm_start_preserves_solution_and_material_gradient(self):
        e = torch.tensor([3.], dtype=torch.float64, requires_grad=True)
        reference = self.simulate(e)
        gradient = torch.autograd.grad(reference[3, 2], e)[0]
        target = self.rest.clone()
        target[1, 0] = 1.1
        warm, _ = matrix_free_step(self.rest, self.tets, self.mass, self.rest,
            torch.zeros_like(self.rest), e, .1, self.fixed, target,
            guess=reference.detach())
        actual_gradient = torch.autograd.grad(warm[3, 2], e)[0]
        torch.testing.assert_close(warm, reference)
        torch.testing.assert_close(actual_gradient, gradient)

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

    def test_solver_progress_preserves_forward_and_adjoint(self):
        target = self.rest.clone()
        target[1, 0] = 1.1
        e = torch.tensor([3.], dtype=torch.float64, requires_grad=True)
        messages = []
        actual, _ = matrix_free_step(self.rest, self.tets, self.mass, self.rest,
            torch.zeros_like(self.rest), e, .1, self.fixed, target, notify=messages.append)
        gradient = torch.autograd.grad(actual[3, 2], e)[0]
        reference = self.simulate(e)
        expected_gradient = torch.autograd.grad(reference[3, 2], e)[0]
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(gradient, expected_gradient)
        self.assertTrue(any("Newton" in message and "gradient norm=" in message for message in messages))
        self.assertTrue(any("sparse solve" in message and "residual=" in message for message in messages))
        self.assertTrue(any("backward adjoint" in message for message in messages))

    def test_initial_rigid_pose_moves_free_nodes_without_inversion(self):
        rotation = torch.tensor([[1., 0, 0], [0, 0, -1.], [0, 1., 0]],
                                dtype=torch.float64)
        target = self.rest @ rotation + torch.tensor([2., -1., 3.])
        e = torch.tensor([8.], dtype=torch.float64, requires_grad=True)
        x, velocity = initial_state(self.rest, self.tets, self.mass, e, .1,
                                    self.fixed, target)
        torch.testing.assert_close(x, target)
        torch.testing.assert_close(velocity, torch.zeros_like(velocity))
        self.assertGreater(float(volumes(x, self.tets).min()), 0)

    def test_adaptive_boundary_motion_preserves_volume_and_material_gradient(self):
        target = self.rest + torch.tensor([0., 0, 1.2])
        mass = self.mass * .001
        e = torch.tensor([8.], dtype=torch.float64, requires_grad=True)
        with self.assertRaises(FEMConvergenceError):
            matrix_free_step(self.rest, self.tets, mass, self.rest,
                             torch.zeros_like(self.rest), e, .1, self.fixed, target)
        def simulate(parameter, halvings=8):
            return adaptive_step(self.rest, self.tets, mass, self.rest,
                torch.zeros_like(self.rest), parameter, .1, self.fixed, target,
                max_halvings=halvings)[0]
        x = simulate(e)
        torch.testing.assert_close(x[self.fixed], target[self.fixed])
        self.assertGreater(float(volumes(x, self.tets).min()), 0)
        gradient = torch.autograd.grad(x[3, 2], e)[0]
        numerical = (simulate(e.detach()+1e-3)[3, 2]
                     - simulate(e.detach()-1e-3)[3, 2])/2e-3
        torch.testing.assert_close(gradient[0], numerical, rtol=1e-2, atol=1e-9)
        with self.assertRaisesRegex(FEMConvergenceError, "adaptive substeps exhausted"):
            simulate(e, halvings=0)

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
