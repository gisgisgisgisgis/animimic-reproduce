import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch


@unittest.skipUnless(importlib.util.find_spec("pytorch3d"), "PyTorch3D unavailable")
class RenderOptimizationTests(unittest.TestCase):
    def test_srgb_display_preserves_mask_and_finite_gradients(self):
        from pytorch3d.utils import ico_sphere
        from animimic.render import Renderer
        sphere = ico_sphere(1)
        vertices = (sphere.verts_packed()*.25).requires_grad_()
        faces = sphere.faces_packed()
        colors = torch.ones_like(vertices)*.05
        linear = Renderer("cpu", 32, 24, lighting="studio")
        display = Renderer("cpu", 32, 24, lighting="studio", srgb=True)
        rgb, mask = linear(vertices, faces, colors)
        actual, actual_mask = display(vertices, faces, colors)
        expected = torch.where(rgb <= .0031308, 12.92*rgb,
                               1.055*rgb.clamp_min(1e-12).pow(1/2.4)-.055)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_mask, mask)
        self.assertGreater(actual[mask>.5].mean(), rgb[mask>.5].mean())
        actual.sum().backward()
        self.assertTrue(torch.isfinite(vertices.grad).all())

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
            self.assertGreaterEqual(renderer.rasterizer.raster_settings.max_faces_per_bin,
                                    2 * len(faces))
            self.assertGreaterEqual(renderer.mask.rasterizer.raster_settings.max_faces_per_bin,
                                    2 * len(faces))
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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_dense_cuda_bins_match_naive_rasterization(self):
        from animimic.render import Renderer
        renderer = Renderer("cuda", 32, 24)
        vertices = torch.tensor([[-.1, -.1, 0.], [.1, -.1, 0.], [0., .1, 0.]],
                                device="cuda")
        # More than the default capacity, concentrated in the same screen bins.
        faces = torch.tensor([[0, 1, 2]], device="cuda").repeat(12000, 1)
        colors = torch.ones_like(vertices) * .5
        with torch.no_grad():
            image, mask = renderer(vertices, faces, colors)
            renderer.rasterizer.raster_settings.bin_size = 0
            renderer.mask.rasterizer.raster_settings.bin_size = 0
            reference, reference_mask = renderer(vertices, faces, colors)
        torch.testing.assert_close(image, reference)
        torch.testing.assert_close(mask, reference_mask)
