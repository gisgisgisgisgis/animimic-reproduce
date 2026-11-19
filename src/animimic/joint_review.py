"""Preview joint animation at the reference video's native frame rate."""
from pathlib import Path

import numpy as np

from .core import save_json


def source_fps(video):
    import imageio.v2 as iio
    reader = iio.get_reader(str(video))
    try:
        fps = float(reader.get_meta_data()["fps"])
    finally:
        reader.close()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Reference video must have a positive frame rate")
    return fps


def interpolate_local(local, times, output_times):
    """Interpolate rotations on SO(3), preserving fixed child offsets/scales."""
    from scipy.spatial.transform import Rotation, Slerp
    times = np.asarray(times, dtype=float)
    output_times = np.asarray(output_times, dtype=float)
    if len(times) != len(local) or len(times) < 2 or not (np.diff(times)>0).all():
        raise ValueError("Joint interpolation requires increasing sample times")
    query = np.clip(output_times, times[0], times[-1])
    scale = np.linalg.norm(local[:, :, :3, :3], axis=-2)
    scale[:, :, -1] *= np.sign(np.linalg.det(local[:, :, :3, :3]))
    if (np.abs(scale) < 1e-12).any():
        raise ValueError("Cannot interpolate a singular joint transform")
    rotation = local[:, :, :3, :3]/scale[:, :, None, :]
    result = np.tile(np.eye(4), (len(query), local.shape[1], 1, 1))
    for joint in range(local.shape[1]):
        interpolated = Slerp(times, Rotation.from_matrix(rotation[:, joint]))(query).as_matrix()
        size = np.stack([np.interp(query, times, scale[:, joint, axis]) for axis in range(3)], -1)
        result[:, joint, :3, :3] = interpolated*size[:, None, :]
        for axis in range(3):
            result[:, joint, axis, 3] = np.interp(query, times, local[:, joint, axis, 3])
    return result.astype(local.dtype)


def write_joint_review(directory, rig, local, renderer, video, indices, fps):
    import imageio.v2 as iio
    import torch
    from PIL import Image
    from .export import export_glb
    from .render import Renderer
    from .rig import fk, lbs

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output_indices = np.arange(int(indices[-1])+1)
    poses = interpolate_local(local, np.asarray(indices)/fps, output_indices/fps)
    portable = {name: rig[name].detach().cpu().numpy() for name in
                ("vertices", "faces", "colors", "weights", "inverse_bind", "parents")}
    export_glb(directory/"joint.glb", portable, poses, fps)
    device = rig["vertices"].device
    poses = torch.as_tensor(poses, device=device)
    parents = rig["parents"].tolist()
    for azimuth in (0, -45, 45):
        view = renderer if azimuth == 0 else Renderer(device,
            **{**renderer.settings, "azimuth": renderer.settings["azimuth"]+azimuth})
        view.bind_material(rig)
        reader = iio.get_reader(str(video)) if azimuth == 0 else None
        try:
            with iio.get_writer(directory/f"joint_{azimuth}.mp4", fps=fps, macro_block_size=1) as writer:
                with torch.no_grad():
                    for index, pose in enumerate(poses):
                        vertices = lbs(rig["vertices"], rig["weights"], fk(pose, parents), rig["inverse_bind"])
                        rgb, _ = view(vertices, rig["faces"], rig["colors"])
                        image = (rgb.clamp(0, 1).cpu().numpy()*255).astype(np.uint8)
                        if reader is not None:
                            reference = Image.fromarray(reader.get_data(index)).resize(view.size[::-1])
                            image = np.concatenate((np.asarray(reference), image), axis=1)
                        writer.append_data(image)
        finally:
            if reader is not None:
                reader.close()
    save_json(directory/"joint-review.json", {"sampled_frames": len(indices),
        "output_frames": len(poses), "fps": fps, "interpolation": "joint rotations: SLERP"})
