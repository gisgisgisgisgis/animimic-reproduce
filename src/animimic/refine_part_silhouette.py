"""Refine a chosen skeleton part using the saved video foreground and joint fit."""
import argparse
import json
from pathlib import Path
from functools import lru_cache

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from .compare_optimizers import artifact
from .core import file_hash, save_json
from .joint_review import source_fps, write_joint_review
from .losses import mesh_edges
from .optimize import tensors
from .render import Renderer
from .rig import fk, lbs


def search_angles(score, count, initial=None):
    """Search in both rotation directions, with progressively smaller steps."""
    angles = np.zeros(count) if initial is None else np.array(initial, dtype=float, copy=True)
    best = score(angles)
    for step in (30., 15., 7.5):
        for joint in range(count):
            choices = [(best, angles)]
            for sign in (-1, 1):
                proposal = angles.copy()
                proposal[joint] += sign*step
                if abs(proposal[joint]) <= 75:
                    choices.append((score(proposal), proposal))
            best, angles = min(choices, key=lambda item: item[0])
    return angles, best


@torch.no_grad()
def refine(rig, renderer, local, masks, joints):
    if len(local) != len(masks) or tuple(masks.shape[-2:]) != renderer.size:
        raise ValueError("Joint poses and video masks must use the same frames and image size")
    parents = rig["parents"].tolist()
    selected = rig["weights"][:, joints].sum(1) > .5
    if not selected.any():
        raise ValueError("Selected joints have no surface vertices")
    edges, rest_lengths = mesh_edges(rig["vertices"], rig["faces"])
    affected = selected[edges].any(dim=1)
    edges, rest_lengths = edges[affected], rest_lengths[affected]
    scale = rest_lengths.clamp_min(rest_lengths.median()*.25).clamp_min(1e-8) if len(edges) else rest_lengths
    previous = np.zeros(len(joints))
    result, report = [], []
    h, w = renderer.size
    for frame, (base, target) in enumerate(zip(local, masks)):
        world = fk(base, parents)
        vertices = lbs(rig["vertices"], rig["weights"], world, rig["inverse_bind"])
        pixels = renderer.project(vertices[selected])
        lower, upper = pixels.amin(dim=0), pixels.amax(dim=0)
        padding = max(12, float((upper-lower).max())*.45)
        x0, y0 = (lower-padding).floor().int().tolist()
        x1, y1 = (upper+padding).ceil().int().tolist()
        crop = (slice(max(0,y0), max(0,min(h,y1))), slice(max(0,x0), max(0,min(w,x1))))
        goal = target[crop]
        if not goal.numel() or not goal.any():
            result.append(base)
            report.append({"applied": False})
            previous[:] = 0
            continue
        # Camera-facing rotation axis in the local coordinates of each bone.
        axis = renderer.camera.R[0, :, 2].expand(len(joints), 3)
        local_axis = torch.linalg.solve(world[joints,:3,:3], axis[...,None]).squeeze(-1)
        local_axis = torch.nn.functional.normalize(local_axis, dim=-1).cpu().numpy()

        def candidate(angles):
            changed = base.clone()
            rotation = Rotation.from_rotvec(local_axis*np.deg2rad(angles)[:,None]).as_matrix()
            changed[joints,:3,:3] = base[joints,:3,:3] @ base.new_tensor(rotation)
            return changed

        base_mask = renderer.mask(renderer.mesh(vertices, rig["faces"], rig["colors"]))[0,...,3]
        before = float((base_mask[crop]-goal).abs().mean())
        # Preserve the rest of the silhouette and discourage implausible skin stretch.
        outside = torch.ones_like(target, dtype=torch.bool)
        outside[crop] = False
        before_outside = (base_mask[outside]-target[outside]).abs().sum()

        @lru_cache(maxsize=None)
        def measure(angles):
            if any(angles):
                pose = candidate(np.asarray(angles))
                v = lbs(rig["vertices"], rig["weights"], fk(pose, parents), rig["inverse_bind"])
                mask = renderer.mask(renderer.mesh(v, rig["faces"], rig["colors"]))[0,...,3]
            else:
                v, mask = vertices, base_mask
            error = (mask[crop]-goal).abs().mean()
            elsewhere = ((mask[outside]-target[outside]).abs().sum()-before_outside).clamp_min(0)/goal.numel()
            length = (v[edges[:,0]]-v[edges[:,1]]).norm(dim=-1)
            stretch = ((length-rest_lengths)/scale).square().mean() if len(edges) else 0.
            return float(error), float(elsewhere + .08*stretch)

        def score(angles):
            error, penalty = measure(tuple(angles))
            temporal = np.square((angles-previous)/75).mean()
            return error + penalty + .02*temporal

        zero = np.zeros(len(joints))
        baseline = score(zero)
        # Try both the original pose and the preceding correction.
        if np.any(previous):
            start = previous if score(previous) < baseline else zero
        else:
            start = zero
        angles, best = search_angles(score, len(joints), start)
        after, _ = measure(tuple(angles))
        applied = best < baseline-.005 and after < before
        result.append(candidate(angles) if applied else base)
        if applied:
            previous = angles
        else:
            previous = zero
        report.append({"applied": bool(applied), "baseline_score": baseline,
                       "candidate_score": best, "initial_mask_error": before,
                       "final_mask_error": after if applied else before,
                       "rotation_degrees": previous.tolist()})
        print(f"[silhouette] fitted frame {frame+1}/{len(local)} | {baseline:.3f} → {best:.3f}", flush=True)
    return torch.stack(result), report


@torch.no_grad()
def write_silhouette_comparisons(output, rig, renderer, original, refined, masks, indices, report):
    parents = rig["parents"].tolist()
    h, w = renderer.size

    def render_mask(pose):
        vertices = lbs(rig["vertices"], rig["weights"], fk(pose, parents), rig["inverse_bind"])
        mesh = renderer.mesh(vertices, rig["faces"], rig["colors"])
        return renderer.mask(mesh)[0, ..., 3] > .5

    for index, base, fixed, target, row in zip(indices, original, refined, masks, report):
        before = render_mask(base)
        after = render_mask(fixed) if row["applied"] else before
        sheet = Image.new("RGB", (w*3, h+22), "white")
        draw = ImageDraw.Draw(sheet)
        for column, (title, mask) in enumerate(zip(
                ("SAM2 target", "Before refinement", "After refinement"),
                (target > .5, before, after))):
            sheet.paste(Image.fromarray(mask.cpu().numpy().astype(np.uint8)*255), (w*column, 22))
            draw.text((w*column+5, 4), title, fill="black")
        sheet.save(output/f"silhouette-{int(index):05d}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run with completed joints stage")
    parser.add_argument("--part", action="append", required=True, help="Repeat to select several rig groups")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--debug-images", action="store_true",
                        help="Export per-frame target, original and refined silhouette comparisons")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("Use the GPU host for the full silhouette search")
    manifest = json.loads((args.run/"manifest.json").read_text())
    cfg = manifest["config"]
    groups = cfg.get("supervision", {}).get("track_groups", {})
    if any(part not in groups for part in args.part):
        parser.error("Each --part must be a group in supervision.track_groups")
    if "joints" not in manifest["stages"]:
        parser.error("Run the grouped-tracks config through --until-stage joints first")
    joints = sorted({joint for part in args.part for joint in groups[part]})
    root = Path.cwd()
    rig_path = artifact(manifest, "prepare", "rig.npz", root)
    supervision = artifact(manifest, "supervision", "supervision.npz", root)
    original = artifact(manifest, "joints", "joint_trajectory.npz", root)
    video = args.run/"reference.mp4"
    if not video.is_file():
        video = artifact(manifest, "video", "reference.mp4", root)
    if file_hash(video) != manifest["stages"]["video"]["artifacts"]["reference.mp4"]:
        raise ValueError("Reference video does not match the saved run")
    with np.load(rig_path) as data:
        rig = tensors(dict(data), args.device)
    with np.load(original) as data:
        local = torch.as_tensor(data["local"], device=args.device)
    with np.load(supervision) as data:
        masks = torch.as_tensor(data["masks"], device=args.device, dtype=local.dtype)
        indices = data["indices"]
    renderer = Renderer(args.device, **cfg.get("camera", {}))
    renderer.bind_material(rig)
    output = args.output or args.run/"silhouette-refinement"
    output.mkdir(parents=True, exist_ok=False)
    result, report = refine(rig, renderer, local, masks, joints)
    for index, row in zip(indices, report):
        row["source_frame"] = int(index)
    save_json(output/"silhouette-refinement.json", {"parts": args.part, "joints": joints, "frames": report})
    np.savez_compressed(output/"poses.npz", local=result.cpu().numpy(), indices=indices)
    if args.debug_images:
        write_silhouette_comparisons(output, rig, renderer, local, result, masks, indices, report)
    write_joint_review(output/"joint-review", rig, result.cpu().numpy(), renderer,
                       video, indices, source_fps(video))
    print(f"[silhouette] Review: {output/'joint-review'/'joint_0.mp4'}")


if __name__ == "__main__":
    main()
