"""Refit joints from saved supervision and manually matched 2D tracks."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .compare_optimizers import artifact
from .core import file_hash, save_json
from .joint_review import source_fps, write_joint_review
from .optimize import fit_joints, tensors
from .render import Renderer
from .track_sampling import track_part_ids


def bind_pixel(value):
    try:
        point, pixel = value.split(":")
        x, y = pixel.split(",")
        return int(point), (int(x), int(y))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use POINT_ID:X,Y") from exc


def anchor_observations(rows, indices, point_ids, start, end, size):
    """Join CSV observations to fitted frames using original video frame indices."""
    xy = np.zeros((len(indices), len(point_ids), 2), dtype=np.float32)
    valid = np.zeros(xy.shape[:-1], dtype=bool)
    frame_slots = {int(frame): i for i, frame in enumerate(indices) if start <= frame <= end}
    point_slots = {point: i for i, point in enumerate(point_ids)}
    seen = set()
    height, width = size
    for row in rows:
        frame, point = int(row["source_frame"]), int(row["point_id"])
        if frame not in frame_slots or point not in point_slots:
            continue
        slot = frame_slots[frame], point_slots[point]
        if slot in seen:
            raise ValueError(f"Duplicate track observation for frame {frame}, point {point}")
        seen.add(slot)
        x, y = float(row["x"]), float(row["y"])
        quality = float(row["visibility"])*float(row["confidence"])
        if quality > .5 and 0 <= x <= width-1 and 0 <= y <= height-1:
            xy[slot], valid[slot] = (x, y), True
    if not valid.any(axis=0).all():
        raise ValueError("Each binding needs a confident observation at a fitted frame in the chosen range")
    return xy, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--bind", required=True, type=bind_pixel, action="append",
                        help="POINT_ID:X,Y, where X,Y are pixels on tracks-00000.png")
    parser.add_argument("--part", required=True, action="append",
                        help="Group name to replace when a matching anchor is visible. Repeat for more groups.")
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--anchor-weight", type=float, default=10.)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    point_ids, pixels = zip(*args.bind)
    if len(set(point_ids)) != len(point_ids) or len(set(pixels)) != len(pixels):
        parser.error("Each point and starting pixel can be bound only once")
    if not 0 < args.start <= args.end:
        parser.error("Require 0 < --start <= --end so frame-zero depth calibration remains available")
    if not np.isfinite(args.anchor_weight) or args.anchor_weight <= 0:
        parser.error("--anchor-weight must be finite and positive")
    if not torch.cuda.is_available():
        parser.error("Use the GPU host to fit joints")
    manifest = json.loads((args.run/"manifest.json").read_text())
    config = manifest["config"]
    groups = config.get("supervision", {}).get("track_groups", {})
    if any(part not in groups for part in args.part):
        parser.error("--part must be an existing supervision.track_groups name")
    settings = config["joints"].copy()
    if settings.get("objective") != "normalized":
        parser.error("Re-anchored observations require the normalized joint objective")
    if args.iterations is not None:
        settings["iterations"] = args.iterations
    settings["loss_weights"] = {**settings.get("loss_weights", {}), "anchor": args.anchor_weight}

    repository = Path.cwd()
    rig_path = artifact(manifest, "prepare", "rig.npz", repository)
    supervision = artifact(manifest, "supervision", "supervision.npz", repository)
    video = args.run/"reference.mp4"
    if not video.is_file():
        video = artifact(manifest, "video", "reference.mp4", repository)
    if file_hash(video) != manifest["stages"]["video"]["artifacts"]["reference.mp4"]:
        raise ValueError("Reference video does not match the saved run")
    with np.load(rig_path) as source:
        raw = dict(source)
    with np.load(supervision) as source:
        data = {k: source[k] for k in source.files if k != "depth_maps"}
    device = "cuda"
    rig = tensors(raw, device)
    renderer = Renderer(device, **config.get("camera", {}))
    renderer.bind_material(rig)
    h, w = renderer.size
    with args.tracks.open(newline="") as stream:
        xy, valid = anchor_observations(csv.DictReader(stream), data["indices"], point_ids,
                                       args.start, args.end, renderer.size)
    anchors = torch.tensor(pixels, device=device)
    if ((anchors < 0) | (anchors >= anchors.new_tensor([w,h]))).any():
        parser.error("--bind pixels must be within the input image")
    with torch.no_grad():
        fragments = renderer.rasterizer(renderer.mesh(rig["vertices"], rig["faces"], rig["colors"]))
        x, y = anchors.T
        tri = fragments.pix_to_face[0, y, x, 0]
        bary = fragments.bary_coords[0, y, x, 0]
    if (tri < 0).any():
        parser.error("A --bind pixel does not hit the starting mesh. Pick a pixel inside the indicated part.")
    names, bound_parts = track_part_ids(raw, tri.cpu().numpy(), bary.cpu().numpy(), groups)
    if {names[part] for part in bound_parts} != set(args.part):
        parser.error("The bound pixels must cover exactly the requested --part groups")
    _, old_parts = track_part_ids(raw, data["triangles"], data["barycentric"], groups)
    excluded = ((valid.astype(int) @ (bound_parts[:, None] == old_parts[None, :])) > 0)
    data.update(exclude_tracks=excluded, anchor_xy=xy, anchor_valid=valid,
                anchor_triangles=tri.cpu().numpy(), anchor_barycentric=bary.cpu().numpy())
    fps = source_fps(video)
    data["frame_times"] = data["indices"]/fps
    target = {key: torch.as_tensor(value, device=device, dtype=torch.long
              if key in ("triangles", "anchor_triangles", "indices") else torch.float32)
              for key, value in data.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output/"anchors.json", {
        "run": str(args.run), "tracks": str(args.tracks), "tracks_sha256": file_hash(args.tracks),
        "part": args.part, "bindings": args.bind, "frames": data["indices"][valid.any(1)].tolist(),
        "observations": int(valid.sum()), "excluded_observations": int(excluded.sum()), "joints": settings,
    })
    print(f"[anchors] {valid.sum()} new 2D observations; {excluded.sum()} old observations disabled", flush=True)
    fit_joints(rig, target, renderer, args.output, settings)
    with np.load(args.output/"joint_trajectory.npz") as result:
        write_joint_review(args.output/"joint-review", rig, result["local"], renderer, video, data["indices"], fps)
    print(f"[anchors] Review: {args.output/'joint-review'/'joint_0.mp4'}", flush=True)


if __name__ == "__main__":
    main()
