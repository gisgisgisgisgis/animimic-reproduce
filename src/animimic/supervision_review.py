"""CPU inspection of saved supervision; never alters optimization targets."""
import argparse
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


def write_review(path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    report = summarize(data)
    report["source"] = str(Path(path).resolve())
    (output/"supervision-review.json").write_text(json.dumps(report, indent=2)+"\n")
    rows = report["per_frame"]
    with (output/"supervision-review.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    indices = np.unique(np.linspace(0, len(rows)-1, min(6, len(rows))).astype(int))
    h, w = data["masks"].shape[1:]
    sheet = Image.new("RGB", (w*2, (h+25)*len(indices)), "white")
    for row, i in enumerate(indices):
        frame = (np.clip(data["frames"][i], 0, 1)*255).astype(np.uint8)
        overlay = frame.copy()
        mask = data["masks"][i] > .5
        overlay[mask] = (.6*overlay[mask]+.4*np.array([0, 220, 80])).astype(np.uint8)
        image = Image.fromarray(overlay)
        draw = ImageDraw.Draw(image)
        for xy, reliable in zip(data["tracks"][i], data["visibility"][i]*data["confidence"][i] > .5):
            x, y = xy
            if 0 <= x < w and 0 <= y < h:
                draw.ellipse((x-1, y-1, x+1, y+1), fill="cyan" if reliable else "red")
        sheet.paste(Image.fromarray(frame), (0, row*(h+25)))
        sheet.paste(image, (w, row*(h+25)))
        ImageDraw.Draw(sheet).text((4, row*(h+25)+h+4),
            f"frame {i} / source {data['indices'][i]} | mask green | reliable tracks cyan; others red", fill="black")
    sheet.save(output/"supervision-review.jpg")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("supervision", help="Path to cached supervision.npz")
    parser.add_argument("--output", default="runs/supervision-review")
    args = parser.parse_args()
    report = write_review(args.supervision, args.output)
    print(f"Inspected {report['frames']} frames, {report['tracks']} tracks: {args.output}")


if __name__ == "__main__":
    main()
