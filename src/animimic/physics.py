"""Tetrahedral preparation, embedding, joint boundary conditions and material fitting."""
import shutil
import subprocess

import numpy as np
import torch

from .core import save_history, save_json
from .fem import adaptive_step, initial_state, FEMConvergenceError, volumes, BACKEND
from .optimize import video_loss
from .progress import Progress


def tetrahedralize(rig, directory, config):
    import meshio
    import trimesh
    mesh = trimesh.Trimesh(rig["vertices"], rig["faces"], process=True)
    proxy = mesh.simplify_quadric_decimation(
        face_count=config.get("target_proxy_faces", 15000)) if len(mesh.faces)>15000 else mesh
    proxy.export(directory/"surface.obj")
    executable = shutil.which(config.get("tetwild_executable", "FloatTetwild_bin"))
    if executable is None:
        raise RuntimeError("fTetWild executable missing")
    subprocess.run([executable, "-i", str(directory/"surface.obj"), "-o",
                    str(directory/"volume.msh"), "-l",
                    str(config.get("tet_edge_ratio", 0.05))], check=True)
    volume = meshio.read(directory/"volume.msh")
    tets = volume.cells_dict["tetra"].copy()
    points = volume.points
    p = points[tets]
    determinant = np.linalg.det(np.stack((p[:, 1]-p[:, 0], p[:, 2]-p[:, 0],
                                          p[:, 3]-p[:, 0]), -1))
    inverted = determinant < 0
    tets[inverted] = tets[inverted][:, [1, 0, 2, 3]]
    if (np.abs(determinant) < 1e-15).any():
        raise ValueError("Degenerate tetrahedra")
    # Nearest-center candidates, then exact barycentric containment or least
    # negative extrapolation. The mapping remains fixed throughout optimization.
    from scipy.spatial import cKDTree
    centers = points[tets].mean(1)
    tree = cKDTree(centers)
    _, candidates = tree.query(rig["vertices"], k=min(32, len(tets)))
    candidates = np.asarray(candidates).reshape(len(rig["vertices"]), -1)
    chosen, beta = [], []
    for vertex, ids in zip(rig["vertices"], candidates):
        tetra = points[tets[ids]]
        matrices = np.stack((tetra[:, 1]-tetra[:, 0], tetra[:, 2]-tetra[:, 0],
                              tetra[:, 3]-tetra[:, 0]), -1)
        tail = np.linalg.solve(matrices, (vertex-tetra[:, 0])[..., None])[..., 0]
        bary = np.c_[1-tail.sum(-1), tail]
        selected = np.maximum(-bary, 0).sum(-1).argmin()
        chosen.append(ids[selected])
        beta.append(bary[selected])
    world = np.linalg.inv(rig["inverse_bind"])
    joints = world[:, :3, 3]
    nearest = tree.query(joints)[1]
    fixed = np.zeros(len(points), bool)
    owner = np.full(len(points), -1)
    # Skip scene ancestors without direct skin influence.
    for j, tet in enumerate(nearest):
        if rig["weights"][:, j].sum() <= 0:
            continue
        for vertex in tets[tet]:
            old = owner[vertex]
            if old < 0 or np.linalg.norm(points[vertex]-joints[j]) < np.linalg.norm(points[vertex]-joints[old]):
                owner[vertex] = j
                fixed[vertex] = True
    if fixed.all() or not fixed.any():
        raise ValueError("Tet resolution leaves no driven/free nodes; adjust edge ratio")
    group = cKDTree(joints).query(centers)[1]
    save_json(directory/"tet_quality.json", {
        "tetrahedra": len(tets), "vertices": len(points),
        "min_rest_volume": float(np.abs(determinant).min()/6),
        "boundary_vertices": int(fixed.sum()),
        "max_barycentric_extrapolation": float(np.maximum(-np.asarray(beta), 0).sum(-1).max()),
    })
    np.savez_compressed(directory/"tetrahedral.npz", rest=points, tets=tets,
                        surface_tets=np.asarray(chosen), barycentric=np.asarray(beta),
                        fixed=fixed, owner=owner, groups=group)
    return ["surface.obj", "volume.msh", "tetrahedral.npz", "tet_quality.json"]


def fit_material(rig, data, joint_world, target, renderer, directory, config, dt):
    if config.get("poisson_ratio", 0.45) != 0.45:
        raise ValueError("This FEM implementation currently fixes Poisson ratio at 0.45")
    BACKEND.set(config.get("backend", "warp"))
    device = rig["vertices"].device
    dtype = torch.float64
    rest = torch.tensor(data["rest"], device=device, dtype=dtype)
    tets = torch.tensor(data["tets"], device=device, dtype=torch.long)
    fixed = torch.tensor(data["fixed"], device=device)
    owner = torch.tensor(data["owner"].clip(0), device=device)
    groups = torch.tensor(data["groups"], device=device)
    embedding = tets[torch.tensor(data["surface_tets"], device=device)]
    bary = torch.tensor(data["barycentric"], device=device, dtype=dtype)
    mass = torch.zeros(len(rest), device=device, dtype=dtype)
    mass.index_add_(0, tets.flatten(), (volumes(rest, tets).abs()[:, None].expand(-1, 4)/4).flatten())
    mass *= config.get("density", 1.)
    inverse = rig["inverse_bind"].double()
    transforms = joint_world.double() @ inverse
    homogeneous = torch.nn.functional.pad(rest, (0, 1), value=1)
    targets = torch.einsum("tvab,vb->tva", transforms[:, owner], homogeneous)[..., :3]
    history = []
    trajectory = None
    for level, iterations in enumerate([config.get("coarse_iterations", 100),
                                         config.get("fine_iterations", 100)]):
        if level:
            # Split each joint cluster along its largest spatial extent.
            centers = rest[tets].mean(1)
            new = groups*2
            for group in groups.unique():
                selected = groups == group
                axis = centers[selected].var(0, unbiased=False).argmax()
                new[selected] += (centers[selected, axis] > centers[selected, axis].median()).long()
            old_values = log_e.detach()
            groups = new
            initial = old_values[torch.arange(int(groups.max())+1, device=device)//2]
        else:
            initial = torch.full((int(groups.max())+1,),
                                 np.log(config.get("young_modulus", 10000)),
                                 device=device, dtype=dtype)
        log_e = initial.clone().requires_grad_()
        optimizer = torch.optim.Adam([log_e], lr=config.get("learning_rate", 1e-3))
        progress = Progress(f"physics {level+1}/2 {'fine' if level else 'coarse'}", iterations)
        for iteration in range(iterations):
            optimizer.zero_grad()
            substeps = config.get("substeps", 4)
            max_halvings = config.get("max_step_halvings", 8)
            progress.detail(f"iteration {iteration+1}/{iterations} | initial pose solve", force=True)
            def initial_report(message):
                progress.detail(f"initial pose | {message}")
            try:
                x, velocity = initial_state(rest, tets, mass, log_e[groups], dt/substeps,
                    fixed, targets[0], max_halvings, initial_report)
            except FEMConvergenceError as error:
                raise FEMConvergenceError(f"Initial pose: {error}") from error
            states = [x]
            for frame in range(1, len(targets)):
                progress.detail(f"FEM simulation frame {frame+1}/{len(targets)}")
                for step in range(substeps):
                    alpha = (step+1)/substeps
                    boundary = targets[frame-1].lerp(targets[frame], alpha)
                    context = f"frame {frame+1}/{len(targets)} | substep {step+1}/{substeps}"
                    def report(message, context=context):
                        progress.detail(f"{context} | {message}")
                    try:
                        x, velocity = adaptive_step(rest, tets, mass, x, velocity,
                            log_e[groups], dt/substeps, fixed, boundary,
                            max_halvings, report)
                    except FEMConvergenceError as error:
                        raise FEMConvergenceError(
                            f"Frame {frame+1}/{len(targets)}, substep {step+1}/{substeps}: {error}") from error
                states.append(x)
            trajectory = torch.stack(states)
            trajectory_log_e = log_e.detach().clone()
            surface = (trajectory[:, embedding]*bary[None, ..., None]).sum(-2).float()
            losses = video_loss(surface, rig, target, renderer, progress=progress)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite material objective")
            progress.detail("backward pass")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([log_e], 1., error_if_nonfinite=True)
            optimizer.step()
            history.append({"level": level, "iteration": iteration,
                            "loss": float(loss.detach())})
            progress.advance(history[-1]["loss"])
            save_history(directory/"physics_losses.json", history)
        np.savez_compressed(directory/"material_checkpoint.npz",
                            log_e=trajectory_log_e.cpu(), groups=groups.cpu())
    if trajectory is None:
        raise ValueError("Material optimization requires positive iteration counts")
    np.savez_compressed(directory/"physics_trajectory.npz",
        vertices=surface.detach().cpu(), tetrahedra=trajectory.detach().cpu(),
        log_e=trajectory_log_e.cpu(), groups=groups.cpu())
    return ["physics_trajectory.npz", "material_checkpoint.npz", "physics_losses.json",
            "physics_losses.csv"]
