"""CPU inspection of saved supervision; never alters optimization targets."""
import argparse
import colorsys
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def summarize(data):
    frames, masks, tracks = (np.asarray(data[k]) for k in ("frames", "masks", "tracks"))
    t, h, w, c = frames.shape
    if (t < 2 or c != 3 or masks.shape != (t, h, w) or tracks.ndim != 3
            or tracks.shape[::2] != (t, 2) or tracks.shape[1] == 0):
        raise ValueError("Expected frames T,H,W,3; masks T,H,W; tracks T,N,2")
    shape = tracks.shape[:2]
    for key in ("visibility", "confidence", "depth"):
        if data[key].shape != shape:
            raise ValueError(f"{key} must have shape {shape}")
    for key in ("frames", "masks", "tracks", "visibility", "confidence", "depth"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"Nonfinite {key}")
    for key in ("masks", "visibility", "confidence"):
        if np.min(data[key]) < 0 or np.max(data[key]) > 1:
            raise ValueError(f"{key} must be in [0, 1]")
    if data["indices"].shape != (t,) or np.any(np.diff(data["indices"]) <= 0):
        raise ValueError("Source frame indices must be strictly increasing")
    foreground = masks > .5
    reliable = data["visibility"]*data["confidence"] > .5
    displacement = np.linalg.norm(tracks-tracks[:1], axis=-1)
    rows = []
    for i in range(t):
        xy = np.rint(tracks[i]).astype(int)
        inside = (xy[:, 0] >= 0)&(xy[:, 0] < w)&(xy[:, 1] >= 0)&(xy[:, 1] < h)
        selected = reliable[i]&inside
        depths = data["depth"][i, selected]
        median = float(np.median(depths)) if depths.size else None
        mad = float(np.median(np.abs(depths-median))) if depths.size else None
        hit = foreground[i, xy[selected, 1], xy[selected, 0]]
        previous = foreground[max(0, i-1)]
        union = (previous | foreground[i]).sum()
        rows.append(dict(frame=i, source_frame=int(data["indices"][i]),
            mask_pixels=int(foreground[i].sum()),
            mask_iou_previous=float((previous & foreground[i]).sum()/union) if union else None,
            reliable_tracks=int(reliable[i].sum()), reliable_in_bounds=int(selected.sum()),
            reliable_inside_mask=float(hit.mean()) if hit.size else None,
            displacement_median_px=float(np.median(displacement[i])),
            reliable_displacement_median_px=float(np.median(displacement[i, selected])) if selected.any() else None,
            depth_median=median, depth_mad=mad,
            depth_mad_inverse=1/max(mad, 1e-6) if mad is not None else None))
    return dict(frames=t, tracks=tracks.shape[1], width=w, height=h,
        reliability_threshold=.5, per_frame=rows,
        note="Metrics flag inconsistencies for inspection; they do not establish a cause or change targets.")


def write_track_frames(data, output, rig, groups):
    from .track_sampling import track_part_ids

    names, part_ids = track_part_ids(rig, data["triangles"], data["barycentric"], groups)
    colors = [(145, 145, 145)] + [
        tuple(round(255 * channel) for channel in colorsys.hsv_to_rgb(i * .618034 % 1, .85, .95))
        for i in range(len(groups))]
    with (output/"track-parts.csv").open("w", newline="") as stream:
        csv_writer = csv.writer(stream)
        csv_writer.writerow(["source_frame", "part", "sampled_points", "in_frame",
                             "high_confidence", "high_confidence_inside_mask"])
        for frame, index in enumerate(data["indices"]):
            image = Image.fromarray((np.clip(data["frames"][frame], 0, 1)*255).astype(np.uint8))
            width, height = image.size
            canvas = Image.new("RGB", (width + 245, max(height, 70 + 23*len(names))), "white")
            canvas.paste(image)
            draw = ImageDraw.Draw(canvas)
            xy = np.rint(data["tracks"][frame]).astype(int)
            in_frame = ((xy >= 0) & (xy < (width, height))).all(axis=1)
            confidence = (data["visibility"][frame]*data["confidence"][frame] > .5) & in_frame
            in_mask = np.zeros(len(part_ids), dtype=bool)
            in_mask[in_frame] = data["masks"][frame, xy[in_frame, 1], xy[in_frame, 0]] > .5
            draw.text((width+10, 8), f"Source frame {index}", fill="black")
            draw.text((width+10, 27), "Solid: confident + inside mask", fill="black")
            draw.text((width+10, 44), "Hollow: check | counts solid / total", fill="black")
            for part, (name, color) in enumerate(zip(names, colors)):
                assigned = part_ids == part
                solid = assigned & confidence & in_mask
                for i in np.flatnonzero(assigned & in_frame):
                    x, y = xy[i]
                    draw.ellipse((x-2, y-2, x+2, y+2), outline=color, fill=color if solid[i] else None)
                row = 64 + 23*part
                draw.rectangle((width+10, row+2, width+20, row+12), fill=color)
                draw.text((width+26, row), f"{name}  {solid.sum()} / {assigned.sum()}", fill="black")
                csv_writer.writerow([index, name, assigned.sum(), (assigned & in_frame).sum(),
                                     (assigned & confidence).sum(), solid.sum()])
            canvas.save(output/f"tracks-{int(index):05d}.png")


def draw_tracks(image, data, frame):
    w, h = image.size
    draw = ImageDraw.Draw(image)
    reliable = data["visibility"][frame]*data["confidence"][frame] > .5
    for (x, y), visible in zip(data["tracks"][frame], reliable):
        if 0 <= x < w and 0 <= y < h:
            draw.ellipse((x-1, y-1, x+1, y+1), fill="cyan" if visible else "red")


def write_review(path, output, *, rig=None, groups=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    report = summarize(data)
    report["source"] = str(Path(path).resolve())
    rows = report["per_frame"]
    with (output/"supervision-review.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    indices = np.unique(np.linspace(0, len(rows)-1, min(6, len(rows))).astype(int))
    h, w = data["masks"].shape[1:]
    depths = data.get("depth_maps")
    if depths is not None:
        finite = np.isfinite(depths)
        near, far = np.percentile(depths[finite], [2, 98]) if finite.any() else (0., 1.)
        report["depth_display_range"] = [float(near), float(far)]
    for i, index in enumerate(data["indices"]):
        Image.fromarray((data["masks"][i] > .5).astype(np.uint8)*255).save(
            output/f"sam-mask-{int(index):05d}.png")
        if depths is not None:
            scaled = np.clip((depths[i]-near)/max(far-near, 1e-8), 0, 1)
            scaled = np.where(finite[i], scaled, 0.)
            Image.fromarray((scaled*255).astype(np.uint8)).save(
                output/f"vggt-depth-{int(index):05d}.png")
    sheet = Image.new("RGB", (w*2, (h+25)*len(indices)), "white")
    for row, i in enumerate(indices):
        frame = (np.clip(data["frames"][i], 0, 1)*255).astype(np.uint8)
        overlay = frame.copy()
        mask = data["masks"][i] > .5
        overlay[mask] = (.6*overlay[mask]+.4*np.array([0, 220, 80])).astype(np.uint8)
        image = Image.fromarray(overlay)
        draw_tracks(image, data, i)
        sheet.paste(Image.fromarray(frame), (0, row*(h+25)))
        sheet.paste(image, (w, row*(h+25)))
        ImageDraw.Draw(sheet).text((4, row*(h+25)+h+4),
            f"frame {i} / source {data['indices'][i]} | mask green | reliable tracks cyan; others red", fill="black")
    sheet.save(output/"supervision-review.jpg")
    if groups:
        write_track_frames(data, output, rig, groups)
    else:
        for i, index in enumerate(data["indices"]):
            frame = (np.clip(data["frames"][i], 0, 1)*255).astype(np.uint8)
            image = Image.fromarray(frame)
            draw_tracks(image, data, i)
            image.save(output/f"tracks-{int(index):05d}.png")
    (output/"supervision-review.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("supervision", help="Path to cached supervision.npz")
    parser.add_argument("--output", default="runs/supervision-review")
    parser.add_argument("--rig", help="Path to the prepared rig.npz for part-colored overlays")
    parser.add_argument("--config", help="YAML containing supervision.track_groups")
    args = parser.parse_args()
    if bool(args.rig) != bool(args.config):
        parser.error("--rig and --config must be supplied together")
    rig, groups = None, None
    if args.rig:
        import yaml
        groups = yaml.safe_load(Path(args.config).read_text())["supervision"]["track_groups"]
        with np.load(args.rig, allow_pickle=False) as saved:
            rig = {key: saved[key] for key in ("faces", "weights")}
    report = write_review(args.supervision, args.output, rig=rig, groups=groups)
    print(f"Inspected {report['frames']} frames, {report['tracks']} tracks: {args.output}")


if __name__ == "__main__":
    main()
