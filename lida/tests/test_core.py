import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from animamimic.camera import Camera  # noqa: E402
from animamimic.fem import TetFEM, lame  # noqa: E402
from animamimic.gltf_rig import RiggedGLB  # noqa: E402
from animamimic.losses import lift_query_points, mean_depth_loss  # noqa: E402
from animamimic.rig import (  # noqa: E402
    Rig, bone_distance, forward_kinematics, identity_pose, linear_blend_skinning, matrix_to_rotation_6d,
    rotation_6d_to_matrix, smooth_skin_weights,
)
from animamimic.stage2 import embed_points  # noqa: E402

torch.set_default_dtype(torch.float32)


def grid_tets(nx, ny, nz, size=1.0):
    pts = np.stack(np.meshgrid(*[np.arange(n + 1) for n in (nx, ny, nz)], indexing="ij"), -1).reshape(-1, 3)
    idx = lambda i, j, k: i * (ny + 1) * (nz + 1) + j * (nz + 1) + k
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = [idx(i + a, j + b, k + d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]
                tets += [[c[0], c[1], c[3], c[7]], [c[0], c[1], c[5], c[7]], [c[0], c[2], c[3], c[7]],
                         [c[0], c[2], c[6], c[7]], [c[0], c[4], c[5], c[7]], [c[0], c[4], c[6], c[7]]]
    return pts.astype(np.float64) * size, np.asarray(tets)


def test_6d_roundtrip():
    q = torch.linalg.qr(torch.randn(16, 3, 3))[0]
    q = q * torch.sign(torch.det(q))[:, None, None]
    assert (rotation_6d_to_matrix(matrix_to_rotation_6d(q)) - q).abs().max() < 1e-5


def test_identity_pose_is_rest():
    rig = Rig(torch.tensor([[0., 0, 0], [0, 1, 0], [0, 2, 0]]), [-1, 0, 1],
              torch.tensor([[0, 1], [1, 2]]), torch.tensor([[.5, .5], [.3, .7]]))
    v = torch.tensor([[.1, .5, 0], [.2, 1.5, .1]])
    r6, t = identity_pose(2, 3, "cpu")
    assert (linear_blend_skinning(rig, v, *forward_kinematics(rig, r6, t)) - v).abs().max() < 1e-6


def test_rotation_pivots_at_joint():
    rig = Rig(torch.tensor([[0., 0, 0], [0, 1, 0]]), [-1, 0], torch.tensor([[1]]), torch.tensor([[1.]]))
    r6, t = identity_pose(1, 2, "cpu")
    rz = torch.tensor([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    r6[0, 1] = matrix_to_rotation_6d(rz)
    out = linear_blend_skinning(rig, torch.tensor([[0., 2, 0]]), *forward_kinematics(rig, r6, t))
    assert (out[0, 0] - torch.tensor([-1., 1, 0])).abs().max() < 1e-6


def test_bone_distance():
    joints = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    rig = Rig(joints, [-1, 0, 1], torch.zeros(4, 1, dtype=torch.long), torch.ones(4, 1))
    verts = torch.tensor([[0.5, 0.25, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.5, 0.75], [2.0, 2.0, 0.0]])
    expected = torch.tensor([0.25, 1.0, 0.75, 2.0 ** 0.5])
    assert (bone_distance(rig, verts) - expected).abs().max() < 1e-6


def _strip(n):
    """A 2 x n strip of quads along x, plus a duplicate of vertex 0 (a UV seam) used by no face."""
    xs = torch.arange(n, dtype=torch.float32)
    verts = torch.cat((torch.stack((xs, torch.zeros(n), torch.zeros(n)), 1), torch.stack((xs, torch.ones(n), torch.zeros(n)), 1)))
    faces = []
    for i in range(n - 1):
        faces += [[i, i + 1, n + i], [i + 1, n + i + 1, n + i]]
    return torch.cat((verts, verts[:1])), torch.tensor(faces)


def test_smooth_skin_weights():
    verts, faces = _strip(10)
    V = verts.shape[0]
    hard = (verts[:, 0] >= 5).long()
    rig = Rig(torch.zeros(2, 3), [-1, 0], torch.stack((hard, 1 - hard), 1), torch.stack((torch.ones(V), torch.zeros(V)), 1))
    out = smooth_skin_weights(rig, verts, faces, steps=20)
    dense = torch.zeros(V, 2).scatter_add_(1, out.skin_idx, out.skin_w)
    assert (dense >= 0).all() and (dense.sum(1) - 1).abs().max() < 1e-6
    assert (dense[-1] - dense[0]).abs().max() < 1e-6  # both copies of the seam vertex agree
    before = (hard[1:10] != hard[:9]).float().max()
    after = (dense[1:10, 1] - dense[:9, 1]).abs().max()
    assert after < 0.5 * before
    assert dense[0, 1] < 0.5 < dense[9, 1]  # each end keeps its own joint
    one = Rig(torch.zeros(2, 3), [-1, 0], torch.zeros(V, 1, dtype=torch.long), torch.ones(V, 1))
    assert (smooth_skin_weights(one, verts, faces, steps=20).skin_w - 1).abs().max() < 1e-6


def test_camera_matches_pytorch3d():
    p3d = pytest.importorskip("pytorch3d.renderer")
    cam = Camera.look_at(2.5, 15.0, 30.0, 40.0, 256)
    R, T = p3d.look_at_view_transform(2.5, 15.0, 30.0)
    assert (cam.R - R[0]).abs().max() < 1e-5 and (cam.T - T[0]).abs().max() < 1e-5
    c = p3d.FoVPerspectiveCameras(R=R, T=T, fov=40.0)
    pts = torch.randn(50, 3) * 0.3
    ref = c.transform_points_screen(pts[None], image_size=((256, 256),))[0]
    xy, z = cam.project(pts)
    assert (xy - ref[:, :2]).abs().max() < 1e-2
    assert (z - c.get_world_to_view_transform().transform_points(pts[None])[0, :, 2]).abs().max() < 1e-4


def test_lift_query_points_recovers_surface():
    verts = torch.tensor([[-.5, -.5, 0.], [.5, -.5, 0.], [0., .5, 0.], [-.5, -.5, .4], [.5, -.5, .4], [0., .5, .4]])
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    cam = Camera.look_at(3.0, 0.0, 0.0, 40.0, 256)
    target = torch.tensor([[0.1, -0.1, 0.4], [-0.1, 0.0, 0.4]])
    q, _ = cam.project(target)
    keep, face, bary = lift_query_points(verts, faces, cam, q)
    assert keep.all() and (face == 1).all()
    assert ((verts[faces[face]] * bary[..., None]).sum(1) - target).abs().max() < 1e-4


@pytest.fixture(scope="module")
def fox():
    path = ROOT / "data/fox/Fox.glb"
    if not path.exists():
        pytest.skip("Fox.glb not downloaded")
    return RiggedGLB(path)


def test_glb_parents_are_topological(fox):
    assert all(p < j for j, p in enumerate(fox.parents))


def test_fox_animation_fits_rig_parameterization(fox):
    """Ground-truth skinning must be exactly representable by our FK + LBS parameters."""
    name = "Walk"
    times = np.linspace(0, fox.animation_duration(name), 6)
    gt = fox.sample_animation(name, times)
    rig, mesh = fox.rig(), fox.mesh()
    assert gt.shape[1] == mesh.verts.shape[0]
    # Solve for each joint's rigid skin transform, then convert to local parameters.
    from scripts.validate_fox import gt_local_pose
    r6, t = gt_local_pose(fox, rig, name, times)
    pred = linear_blend_skinning(rig, mesh.verts, *forward_kinematics(rig, r6, t))
    err = (pred - gt).norm(dim=-1)
    assert err.max() < 2e-3, f"max error {err.max():.5f}"


@pytest.fixture(scope="module")
def beam():
    X, tets = grid_tets(4, 1, 1, 0.1)
    fem = TetFEM(X, tets, density=1000.0, nu=0.3)
    return fem


def test_rest_state_has_no_force(beam):
    E = torch.full((beam.tets.shape[0],), 1e5, dtype=torch.float64)
    st = beam.state(beam.X)
    assert beam.elastic_grad(st, *lame(E, beam.nu)).abs().max() < 1e-8


def test_elastic_gradient_matches_energy(beam):
    E = torch.full((beam.tets.shape[0],), 1e5, dtype=torch.float64)
    mu, lam = lame(E, beam.nu)
    x = beam.X + 0.01 * torch.randn_like(beam.X)
    g = beam.elastic_grad(beam.state(x), mu, lam)
    d = torch.randn_like(x)
    eps = 1e-6
    fd = (beam.energy(beam.state(x + eps * d), mu, lam) - beam.energy(beam.state(x - eps * d), mu, lam)) / (2 * eps)
    assert abs(float(fd - (g * d).sum())) < 1e-4 * abs(float(fd)) + 1e-8


def test_hessian_matches_gradient_difference(beam):
    E = torch.full((beam.tets.shape[0],), 1e5, dtype=torch.float64)
    mu, lam = lame(E, beam.nu)
    x = beam.X + 0.01 * torch.randn_like(beam.X)
    st = beam.state(x)
    d = torch.randn_like(x)
    eps = 1e-6
    fd = (beam.elastic_grad(beam.state(x + eps * d), mu, lam) - beam.elastic_grad(beam.state(x - eps * d), mu, lam)) / (2 * eps)
    hv = beam.hess_vec(st, d, mu, lam)
    assert (fd - hv).norm() < 1e-4 * fd.norm()


def _beam_rollout(fem, E, steps=8, h=1 / 48):
    pinned = torch.where(fem.X[:, 0] < 1e-9)[0]
    u = torch.linspace(0, 1, steps + 1, dtype=torch.float64)[:, None, None]
    shift = torch.tensor([[[0.0, 0.08, 0.0]]], dtype=torch.float64)
    traj = fem.X[pinned][None] + u * shift
    xs = fem.simulate(fem.X.clone(), torch.zeros_like(fem.X), E, pinned, traj, h)
    return xs, pinned, h


def test_adjoint_gradient_matches_finite_difference(beam):
    E = torch.full((beam.tets.shape[0],), 2e4, dtype=torch.float64)
    target = torch.randn_like(beam.X) * 0.002 + beam.X
    xs, pinned, h = _beam_rollout(beam, E)

    def loss_of(xs_):
        return ((xs_[[4, 8]] - target) ** 2).sum()

    dL = torch.zeros_like(xs)
    dL[[4, 8]] = 2.0 * (xs[[4, 8]] - target)
    grad = beam.adjoint_gradient(xs, E, pinned, h, dL)

    chosen = torch.randn_like(E)
    eps = 1e-3 * E.mean()
    plus = loss_of(_beam_rollout(beam, E + eps * chosen / chosen.norm())[0])
    minus = loss_of(_beam_rollout(beam, E - eps * chosen / chosen.norm())[0])
    fd = float((plus - minus) / (2 * eps))
    ad = float((grad * chosen / chosen.norm()).sum())
    assert abs(fd - ad) < 0.02 * abs(fd), f"fd={fd:.6e} ad={ad:.6e}"


def test_material_recovery_on_beam():
    """Known stiffness must be recovered from the motion alone."""
    X, tets = grid_tets(4, 1, 1, 0.1)
    fem = TetFEM(X, tets, density=1000.0, nu=0.3)
    centroid_x = fem.X[fem.tets].mean(1)[:, 0]
    cluster = (centroid_x > centroid_x.median()).long()
    true_log = torch.tensor([np.log(2e4), np.log(6e4)], dtype=torch.float64)
    target, pinned, h = _beam_rollout(fem, torch.exp(true_log)[cluster], steps=12)

    theta = torch.tensor([np.log(1e5), np.log(1e5)], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.08)
    # The free end barely affects the motion, so its stiffness converges slowly.
    sched = torch.optim.lr_scheduler.StepLR(opt, 300, 0.7)
    for _ in range(1500):
        E = torch.exp(theta.detach())[cluster]
        xs, _, _ = _beam_rollout(fem, E, steps=12)
        dL = 2.0 * (xs - target) * 1e3
        grad = fem.adjoint_gradient(xs, E, pinned, h, dL)
        g = torch.zeros_like(theta).index_add_(0, cluster, E * grad)
        opt.zero_grad()
        theta.grad = g
        opt.step()
        sched.step()
    err = (torch.exp(theta.detach()) / torch.exp(true_log) - 1).abs().max()
    assert err < 0.05, f"relative stiffness error {err:.3f}, got {torch.exp(theta).tolist()}"


def test_embed_points_inside_tet():
    X, tets = grid_tets(2, 2, 2, 0.5)
    pts = np.array([[0.31, 0.42, 0.13], [0.77, 0.2, 0.9]])
    t, b = embed_points(pts, X, tets)
    recon = (X[tets[t]] * b[..., None]).sum(1)
    assert np.abs(recon - pts).max() < 1e-9


def test_mean_depth_loss_ignores_shape_and_hidden_points():
    target = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    visible = torch.tensor([[True, True, True], [True, True, False]])
    # Same mean per frame, different shape: no loss.
    same_mean = torch.tensor([[2.0, 2.0, 2.0], [1.5, 1.5, 9.0]])
    assert float(mean_depth_loss(same_mean, target, visible)) < 1e-6
    # Whole body 0.5 closer in frame 0 only: loss is half of 0.5, averaged over 2 frames.
    closer = target.clone()
    closer[0] -= 0.5
    assert abs(float(mean_depth_loss(closer, target, visible)) - 0.25) < 1e-6
