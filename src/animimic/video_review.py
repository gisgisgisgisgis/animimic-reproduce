"""Collect review artifacts without invoking supervision or optimization."""
from pathlib import Path
import shutil

from PIL import Image, ImageDraw

from .core import save_json


def write_review(canonical, video, output, config):
    import imageio.v2 as iio

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for source, name in ((canonical, "canonical.png"), (video, "reference.mp4")):
        if Path(source).resolve() != (output/name).resolve():
            shutil.copy2(source, output/name)
    sheet = Image.new("RGB", (960, 612), "white")
    draw = ImageDraw.Draw(sheet)

    def cell(image, index, title):
        preview = image.convert("RGB")
        preview.thumbnail((320, 180), Image.Resampling.LANCZOS)
        x, y = index % 3 * 320, index // 3 * 204
        sheet.paste(preview, (x+(320-preview.width)//2, y+24+(180-preview.height)//2))
        draw.text((x+8, y+5), title, fill="black")

    with Image.open(canonical) as image:
        cell(image, 0, "Input image")
    reader = iio.get_reader(str(video))
    try:
        count = reader.count_frames()
        if count < 2:
            raise ValueError("Video review requires at least two frames")
        indices = [round(i*(count-1)/7) for i in range(8)]
        for cell_index, frame_index in enumerate(indices, 1):
            cell(Image.fromarray(reader.get_data(frame_index)), cell_index,
                 f"Frame {frame_index}/{count-1}")
        metadata = reader.get_meta_data()
    finally:
        reader.close()
    sheet.save(output/"video-review.jpg", quality=95)
    save_json(output/"video-review.json", {
        "video_config": config, "frames": count, "fps": metadata.get("fps"),
        "sampled_frames": indices, "status": "needs_visual_review",
        "checks": ["Exactly one animal, matching the input image",
                   "Stable appearance, anatomy, background and camera",
                   "Hooves and torso stay in place; intended head motion is visible"],
    })
