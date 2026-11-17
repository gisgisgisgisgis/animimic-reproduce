import tempfile
import unittest
from pathlib import Path

import numpy as np

from animimic.export import export_glb
from animimic.glb import load_rig, inspect_skin
from animimic.glb import rotate_rig


class ExportTests(unittest.TestCase):
    def test_canonical_rotation_preserves_skin_bind(self):
        import torch
        from animimic.rig import fk, lbs
        rig = dict(vertices=np.array([[.2, .1, .3]]), parents=np.array([-1]),
                   local=np.eye(4)[None], inverse_bind=np.eye(4)[None],
                   weights=np.ones((1, 1)), colors=np.ones((1, 3)))
        rotated = rotate_rig(rig, [-90, 0, 0])
        world = fk(torch.tensor(rotated["local"]), [-1])
        result = lbs(torch.tensor(rotated["vertices"]), torch.tensor(rotated["weights"]),
                     world, torch.tensor(rotated["inverse_bind"]))
        np.testing.assert_allclose(result.numpy(), rotated["vertices"], atol=1e-8)
        np.testing.assert_allclose(rotated["vertices"], [[.2, .3, -.1]], atol=1e-8)

    def test_round_trip_rest_geometry_and_skin(self):
        vertices = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0]])
        rig = dict(vertices=vertices, faces=np.array([[0, 1, 2]]),
                   weights=np.ones((3, 1)), colors=np.ones((3, 3)),
                   local=np.eye(4)[None], inverse_bind=np.eye(4)[None],
                   parents=np.array([-1]))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"test.glb"
            export_glb(path, rig, np.eye(4)[None, None].repeat(2, 0), 24)
            self.assertEqual(inspect_skin(path)["skins"][0]["joints"], 1)
            imported = load_rig(path)
            np.testing.assert_allclose(imported["weights"].sum(-1), 1)
            expected = (vertices-.5*np.array([1., 1, 0]))/np.sqrt(2)
            np.testing.assert_allclose(imported["vertices"], expected)
