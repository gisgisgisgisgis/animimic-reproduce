"""Input-view metrics, reported without claiming paper-equivalent benchmarks."""
import torch


@torch.no_grad()
def metrics(vertices, rig, target, renderer):
    masks = []
    for frame in vertices:
        _, alpha = renderer(frame, rig["faces"], rig["colors"])
        masks.append(alpha>.5)
    mask = torch.stack(masks)
    truth = target["masks"]>.5
    intersection = (mask & truth).sum((-2, -1)).float()
    union = (mask | truth).sum((-2, -1)).clamp_min(1).float()
    ids = rig["faces"][target["triangles"]]
    points = (vertices[:, ids]*target["barycentric"][None, ..., None]).sum(-2)
    projected = torch.stack([renderer.project(p) for p in points])
    error = (projected-target["tracks"]).norm(dim=-1)
    weight = target["visibility"]*target["confidence"]
    acceleration = vertices[2:]-2*vertices[1:-1]+vertices[:-2]
    return {"mask_iou": float((intersection/union).mean()),
            "visible_track_error_pixels": float((error*weight).sum()/weight.sum().clamp_min(1)),
            "mean_vertex_acceleration": float(acceleration.norm(dim=-1).mean())
            if len(vertices)>2 else 0.}
