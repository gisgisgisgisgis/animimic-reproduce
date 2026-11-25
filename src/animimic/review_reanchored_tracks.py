"""Review 2D tracks started from a chosen frame, in both time directions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def track_both_ways(frames, anchor_index, track):
    """Return observations in chronological order, with the anchor included once."""
    before_count = anchor_index
    after_count = len(frames) - anchor_index - 1
    if not before_count and not after_count:
        raise ValueError("The review needs at least one frame besides the anchor")
    backward = track(frames[:anchor_index + 1][::-1].copy()) if before_count else None
    forward = track(frames[anchor_index:].copy()) if after_count else None
    if backward is None:
        return forward
    if forward is None:
        return tuple(values[::-1].copy() for values in backward)
    return tuple(np.concatenate((past[:0:-1], future), axis=0)
                 for past, future in zip(backward, forward))


def pixel(value):
    try:
        x, y = value.split(",")
        return int(x), int(y)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use a point in x,y pixel coordinates") from exc


def main():
    import imageio.v2 as iio
    import torch

    from .models import track_frames

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Completed run with manifest.json and reference.mp4")
    parser.add_argument("--anchor", required=True, type=int, help="Zero-based source video frame")
    parser.add_argument("--point", required=True, type=pixel, action="append",
                        help="Repeat for each x,y pixel on the resized anchor frame")
    parser.add_argument("--radius", type=int, default=15, help="Source frames to inspect on each side")
    parser.add_argument("--stride", type=int, default=1, help="Gap in source frames between tracker inputs")
    parser.add_argument("--output", type=Path, help="Defaults to RUN/reanchored-tracks/ANCHOR")
    args = parser.parse_args()
    if args.radius < 1:
        parser.error("--radius must be positive")
    if not 1 <= args.stride <= args.radius:
        parser.error("--stride must be positive and no larger than --radius")

    manifest = json.loads((args.run/"manifest.json").read_text())
    config = manifest["config"]
    camera = config.get("camera", {})
    width, height = camera.get("width", 512), camera.get("height", 288)
    if any(not (0 <= x < width and 0 <= y < height) for x, y in args.point):
        parser.error(f"--point must be inside the {width}x{height} image")
    if not torch.cuda.is_available():
        parser.error("AllTracker requires CUDA in this project. Run this on the GPU host.")
    video = args.run/"reference.mp4"
    if not video.is_file():
        video = Path(manifest["stages"]["video"]["directory"])/"reference.mp4"
    with iio.get_reader(str(video)) as reader:
        count = reader.count_frames()
        if not 0 <= args.anchor < count:
            parser.error(f"--anchor must be in [0, {count-1}]")
        start = max(0, args.anchor-args.radius)
        end = min(count, args.anchor+args.radius+1)
        before = list(range(args.anchor-args.stride, start-1, -args.stride))[::-1]
        indices = before + list(range(args.anchor, end, args.stride))
        if len(indices) < 2:
            parser.error("The selected stride and video bounds leave only the anchor frame")
        frames = np.stack([np.asarray(Image.fromarray(reader.get_data(i)).resize((width, height)))
                           for i in indices])

    points = torch.tensor(args.point, dtype=torch.float32)
    tracks, visibility, confidence = track_both_ways(
        frames, len(before),
        lambda clip: track_frames(clip, points, config.get("supervision", {})))
    directory = f"{args.anchor:05d}"
    if args.stride != 1:
        directory += f"-stride{args.stride}"
    output = args.output or args.run/"reanchored-tracks"/directory
    output.mkdir(parents=True, exist_ok=True)
    with (output/"tracks.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_frame", "point_id", "x", "y", "visibility", "confidence"])
        for i, (frame_id, frame) in enumerate(zip(indices, frames)):
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image)
            draw.text((8, 8), f"Source frame {frame_id} | anchor {args.anchor} | stride {args.stride}", fill="black")
            for j, (x, y) in enumerate(tracks[i]):
                vis, conf = float(visibility[i, j]), float(confidence[i, j])
                writer.writerow([frame_id, j+1, float(x), float(y), vis, conf])
                if 0 <= x < width and 0 <= y < height:
                    draw.ellipse((x-3, y-3, x+3, y+3), outline="black",
                                 fill="cyan" if vis*conf > .5 else "orange")
                    draw.text((x+5, y-5), str(j+1), fill="black")
            image.save(output/f"tracks-{frame_id:05d}.png")
    print(f"[review] AllTracker re-anchored overlays: {output}", flush=True)


if __name__ == "__main__":
    main()
