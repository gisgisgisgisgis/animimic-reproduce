import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch


@unittest.skipUnless(importlib.util.find_spec("pytorch3d"), "PyTorch3D unavailable")
class RenderOptimizationTests(unittest.TestCase):
    def test_real_render_and_joint_optimizer(self):
        from pytorch3d.utils import ico_sphere
        from animimic.render import Renderer
        from animimic.optimize import fit_joints
        sphere = ico_sphere(1)
        vertices, faces = sphere.verts_packed()*.25, sphere.faces_packed()
        rig = dict(vertices=vertices, faces=faces, colors=torch.ones_like(vertices)*.5,
                   weights=torch.ones(len(vertices), 1), local=torch.eye(4)[None],
                   inverse_bind=torch.eye(4)[None], parents=torch.tensor([-1]))
        renderer = Renderer("cpu", 32, 24)
        with torch.no_grad():
            tri, beta, _ = renderer.surface_samples(vertices, faces, rig["colors"], 16)
            image, mask = renderer(vertices, faces, rig["colors"])
            points = (vertices[faces[tri]]*beta[..., None]).sum(-2)
            tracks = renderer.project(points)[None].repeat(2, 1, 1)
            depth = renderer.depth(points)[None].repeat(2, 1)
        target = dict(frames=image[None].repeat(2, 1, 1, 1),
                      masks=mask[None].repeat(2, 1, 1), triangles=tri, barycentric=beta,
                      tracks=tracks, depth=depth, visibility=torch.ones(2, len(tri)),
                      confidence=torch.ones(2, len(tri)))
        with tempfile.TemporaryDirectory() as directory:
            files = fit_joints(rig, target, renderer, Path(directory),
                dict(root_iterations=1, geometry_iterations=1, full_iterations=1))
            for name in files:
                self.assertTrue((Path(directory)/name).is_file())
