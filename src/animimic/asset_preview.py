"""CPU contact sheets to review geometry, appearance and rig before generation."""
from pathlib import Path


def preview_asset(path, output, name, rotation=None):
    import torch
    from PIL import Image, ImageDraw
    from .glb import load_rig
    from .optimize import tensors
    from .render import Renderer

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    raw = load_rig(path)
    if rotation:
        from .glb import rotate_rig
        raw = rotate_rig(raw, rotation)
    rig = tensors(raw, "cpu")
    sheet = Image.new("RGB", (768, 280), "white")
    with torch.no_grad():
        for column, angle in enumerate((0, 60, 120)):
            renderer = Renderer("cpu", 256, 240, angle)
            rgb, _ = renderer(rig["vertices"], rig["faces"], rig["colors"])
            image = Image.fromarray((rgb.clamp(0, 1).numpy()*255).astype("uint8"))
            sheet.paste(image, (column*256, 30))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), f"{name}: {len(raw['faces'])} triangles, {len(raw['parents'])} rig nodes",
              fill="black")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    result = directory/(name+".png")
    sheet.save(result)
    return result
