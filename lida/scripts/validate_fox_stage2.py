"""Integration check for Stage 2 on the fox, starting from a finished Stage 1 run.

The fox's ground-truth motion has no soft-body dynamics, so Stage 2 cannot be
expected to beat Stage 1 here. This run confirms that tet meshing, surface
embedding, joint-driven simulation, the adjoint and the renderer work together.
Physical correctness is covered by the finite-difference and stiffness-recovery
unit tests in tests/test_core.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytetwild
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from animamimic.device import pick_device  # noqa: E402
from animamimic.rig import forward_kinematics  # noqa: E402
from animamimic.stage2 import Stage2Config, run_stage2  # noqa: E402
from scripts.validate_fox import build_fox_case, make_parser  # noqa: E402


def main():
    ap = make_parser()
    ap.add_argument("--stage1", required=True, help="stage1.npz from validate_fox.py")
    ap.add_argument("--edge_length_fac", type=float, default=0.1)
    ap.add_argument("--stage2_iters", type=int, nargs="+", default=[4, 3])
    args = ap.parse_args()

    torch.manual_seed(0)
    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seq, gt_verts = build_fox_case(args, device)

    data = np.load(args.stage1)
    r6d = torch.from_numpy(data["r6d"]).to(device)
    t = torch.from_numpy(data["t"]).to(device)
    A_r, A_t = forward_kinematics(seq.rig, r6d, t)
    stage1 = {"A_r": A_r, "A_t": A_t}

    verts, tets = pytetwild.tetrahedralize(
        seq.mesh.verts.cpu().numpy().astype(np.float64), seq.mesh.faces.cpu().numpy().astype(np.int32),
        optimize=True, edge_length_fac=args.edge_length_fac,
    )
    verts, tets = np.asarray(verts, np.float64), np.asarray(tets, np.int64)
    print(f"tet mesh: {len(verts)} nodes, {len(tets)} tets")

    cfg = Stage2Config(iters_per_level=args.stage2_iters, batch_frames=args.batch_frames)
    res = run_stage2(seq, stage1, verts, tets, cfg, device, args.backend)

    diag = float((seq.mesh.verts.max(0).values - seq.mesh.verts.min(0).values).norm())
    stage1_verts = torch.from_numpy(data["verts"]).to(device)
    report = {
        "tet_nodes": len(verts),
        "tets": len(tets),
        "clusters_final": int(res["cluster"].max()) + 1,
        "E_min": float(res["E"].min()),
        "E_max": float(res["E"].max()),
        "stage1_vertex_error_pct": 100.0 * float((stage1_verts - gt_verts).norm(dim=-1).mean()) / diag,
        "stage2_vertex_error_pct": 100.0 * float((res["verts"] - gt_verts).norm(dim=-1).mean()) / diag,
        "loss_first": res["history"][0]["loss"],
        "loss_last": res["history"][-1]["loss"],
    }
    with open(out / "stage2_report.json", "w") as f:
        json.dump({"report": report, "history": res["history"]}, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
