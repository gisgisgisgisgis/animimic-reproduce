import unittest
import importlib.util


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
class RigTests(unittest.TestCase):
    def test_rest_pose_and_gradients(self):
        import torch
        from animimic.rig import fk, lbs
        local = torch.eye(4).repeat(2, 1, 1)
        local[1, 0, 3] = 1
        world = fk(local, [-1, 0])
        inverse = torch.linalg.inv(world)
        vertices = torch.tensor([[1.5, 0., 0.]], requires_grad=True)
        result = lbs(vertices, torch.tensor([[0., 1.]]), world, inverse)
        torch.testing.assert_close(result, vertices)
        result.sum().backward()
        torch.testing.assert_close(vertices.grad, torch.ones_like(vertices))
