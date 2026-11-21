"""Volumetric mesh for Stage 2 using fTetWild, via pytetwild. The paper uses TetWild."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from animamimic.gltf_rig import RiggedGLB  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--edge_length_fac", type=float, default=0.08)
    ap.add_argument("--optimize", action="store_true", default=True)
    args = ap.parse_args()

    import pytetwild

    seq = Path(args.seq)
    mesh = RiggedGLB(seq / "rigged.glb").mesh()
    verts, tets = pytetwild.tetrahedralize(
        mesh.verts.numpy().astype(np.float64), mesh.faces.numpy().astype(np.int32),
        optimize=args.optimize, edge_length_fac=args.edge_length_fac,
    )
    np.savez(seq / "tet.npz", verts=np.asarray(verts, np.float64), tets=np.asarray(tets, np.int64))
    print(f"Wrote {len(verts)} nodes and {len(tets)} tets to {seq}/tet.npz")


if __name__ == "__main__":
    main()
