"""Locate mesh damage in a Stage 1 result: seams, edge stretch, joint motion, and a known-pose skinning test."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from animamimic.gltf_rig import RiggedGLB  # noqa: E402
from animamimic.rig import forward_kinematics, identity_pose, linear_blend_skinning, matrix_to_rotation_6d  # noqa: E402


def weld_groups(verts, tol=1e-5):
    _, group = np.unique(np.round(verts / tol).astype(np.int64), axis=0, return_inverse=True)
    return group.reshape(-1)


def component_sizes(n, faces):
    i = np.concatenate((faces[:, 0], faces[:, 1]))
    j = np.concatenate((faces[:, 1], faces[:, 2]))
    g = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    count, label = connected_components(g, directed=False)
    return count, np.sort(np.bincount(label))[::-1]


def unique_edges(faces):
    e = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    return np.unique(np.sort(e, axis=1), axis=0)


def dense_weights(rig, n):
    w = np.zeros((n, rig.num_joints), np.float32)
    idx, val = rig.skin_idx.numpy(), rig.skin_w.numpy()
    for m in range(idx.shape[1]):
        np.add.at(w, (np.arange(n), idx[:, m]), val[:, m])
    return w


def stretch(verts, rest_len, edges):
    """verts: (T, V, 3). Edge length over rest length, (T, E)."""
    d = verts[:, edges[:, 0]] - verts[:, edges[:, 1]]
    return d.norm(dim=-1) / rest_len


def axis_angle_matrix(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    if axis == 0:
        return [[1, 0, 0], [0, c, -s], [0, s, c]]
    if axis == 1:
        return [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def known_pose(rig, mesh_verts, rest_len, edges, degrees):
    """Rotate each non-root joint alone by a known angle and record how much the worst edges change."""
    rows = []
    for j, parent in enumerate(rig.parents):
        if parent < 0:
            continue
        for axis in range(3):
            r6d, t = identity_pose(1, rig.num_joints, "cpu")
            r = torch.tensor(axis_angle_matrix(axis, math.radians(degrees)), dtype=r6d.dtype)
            r6d[0, j] = matrix_to_rotation_6d(r)
            A_r, A_t = forward_kinematics(rig, r6d, t)
            s = stretch(linear_blend_skinning(rig, mesh_verts, A_r, A_t), rest_len, edges)[0]
            rows.append((j, axis, float(s.max()), float(s.min()), int(((s > 1.5) | (s < 0.67)).sum())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--stage1", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    glb = RiggedGLB(args.glb)
    mesh, rig = glb.mesh(), glb.rig()
    verts, faces = mesh.verts.numpy(), mesh.faces.numpy()
    n = len(verts)
    report = {"V": n, "F": len(faces), "joints": rig.num_joints}

    # Structure
    group = weld_groups(verts)
    report["unique_positions"] = int(group.max()) + 1
    report["raw_components"], sizes = component_sizes(n, faces)
    report["raw_component_sizes_top"] = sizes[:8].tolist()
    welded = group[faces]
    report["welded_components"], sizes = component_sizes(group.max() + 1, welded)
    report["welded_component_sizes_top"] = sizes[:8].tolist()

    # Weight disagreement between vertices that share a position.
    w = dense_weights(rig, n)
    first = np.full(group.max() + 1, -1)
    for v in range(n):
        if first[group[v]] < 0:
            first[group[v]] = v
    dup = np.array([v for v in range(n) if first[group[v]] != v])
    report["seam_vertices"] = int(len(dup))
    if len(dup):
        l1 = np.abs(w[dup] - w[first[group[dup]]]).sum(1)
        report["seam_weight_L1"] = {
            "median": float(np.median(l1)), "p99": float(np.percentile(l1, 99)), "max": float(l1.max()),
            "frac_gt_0.1": float((l1 > 0.1).mean()),
        }

    # Weight smoothness along edges, per unit of edge length.
    edges = unique_edges(faces)
    rest_len = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
    keep = rest_len > 1e-6
    report["zero_length_edges"] = int((~keep).sum())
    edges, rest_len = edges[keep], rest_len[keep]
    jump = np.abs(w[edges[:, 0]] - w[edges[:, 1]]).sum(1)
    report["edge_weight_L1"] = {
        "median": float(np.median(jump)), "p99": float(np.percentile(jump, 99)), "p999": float(np.percentile(jump, 99.9)),
        "max": float(jump.max()), "frac_gt_0.5": float((jump > 0.5).mean()), "frac_gt_1.0": float((jump > 1.0).mean()),
    }
    report["rest_edge_len"] = {"median": float(np.median(rest_len)), "max": float(rest_len.max())}
    edges_t, rest_t = torch.from_numpy(edges), torch.from_numpy(rest_len).float()

    # Known-pose test: damage from clean rotations of one joint.
    for deg in (30, 60):
        rows = known_pose(rig, mesh.verts, rest_t, edges_t, deg)
        rows.sort(key=lambda r: -r[2])
        report[f"known_pose_{deg}deg"] = {
            "max_stretch_median_over_joints": float(np.median([r[2] for r in rows])),
            "max_stretch_worst": rows[0][2],
            "bad_edges_median": float(np.median([r[4] for r in rows])),
            "bad_edges_worst": max(r[4] for r in rows),
            "worst_rows(joint,axis,max,min,bad)": rows[:5],
        }

    if args.stage1:
        data = np.load(args.stage1)
        r6d, t = torch.from_numpy(data["r6d"]), torch.from_numpy(data["t"])
        posed = torch.from_numpy(data["verts"])
        A_r, A_t = forward_kinematics(rig, r6d, t)
        rebuilt = linear_blend_skinning(rig, mesh.verts, A_r, A_t)
        report["rebuilt_vs_saved_max"] = float((rebuilt - posed).norm(dim=-1).max())

        def per_frame(v):
            s = stretch(v, rest_t, edges_t)
            return {
                "max": s.max(1).values.tolist(),
                "p999": torch.quantile(s, 0.999, dim=1).tolist() if s.shape[1] < 16_000_000 else None,
                "bad_frac": ((s > 1.5) | (s < 0.67)).float().mean(1).tolist(),
            }

        full = per_frame(posed)
        report["stage1_stretch"] = {k: [round(x, 4) for x in full[k]] for k in full}

        # The same rotations with bone translation removed, to see what the translations contribute.
        t_root = t.clone()
        non_root = [j for j, p in enumerate(rig.parents) if p >= 0]
        t_root[:, non_root] = 0.0
        A_r0, A_t0 = forward_kinematics(rig, r6d, t_root)
        rot_only = per_frame(linear_blend_skinning(rig, mesh.verts, A_r0, A_t0))
        report["rotation_only_stretch"] = {k: [round(x, 4) for x in rot_only[k]] for k in rot_only}

        # Per-joint motion.
        offsets = rig.offsets()
        bone = offsets.norm(dim=-1)
        R = forward_kinematics.__globals__["rotation_6d_to_matrix"](r6d)
        ref = R[0].transpose(-1, -2) @ R  # relative to frame 0
        angle = torch.rad2deg(torch.acos(((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)))
        table = []
        for j, parent in enumerate(rig.parents):
            table.append({
                "joint": j, "parent": parent, "bone_len": round(float(bone[j]), 4),
                "rest_xyz": [round(float(x), 3) for x in rig.joints[j]],
                "angle_max": round(float(angle[:, j].max()), 1),
                "angle_at_T": round(float(angle[-1, j]), 1),
                "t_max": round(float(t[:, j].norm(dim=-1).max()), 4),
                "t_over_bone_max": round(float(t[:, j].norm(dim=-1).max() / bone[j].clamp_min(1e-4)), 2),
                "t_at_clamp_axes": int((t[:, j].abs() > 0.049).any(0).sum()),
            })
        report["joints_table"] = table

        # Which joints the badly stretched edges belong to, on the worst frame.
        s = stretch(posed, rest_t, edges_t)
        worst = int(s.max(1).values.argmax())
        report["worst_frame"] = worst
        bad = ((s[worst] > 1.5) | (s[worst] < 0.67)).numpy()
        report["worst_frame_bad_edges"] = int(bad.sum())
        if bad.any():
            ends = np.concatenate((edges[bad, 0], edges[bad, 1]))
            dominant = np.argmax(w[ends], axis=1)
            count = np.bincount(dominant, minlength=rig.num_joints)
            report["worst_frame_dominant_joints"] = [(int(i), int(count[i])) for i in np.argsort(-count)[:8] if count[i]]

    text = json.dumps(report, indent=1, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
