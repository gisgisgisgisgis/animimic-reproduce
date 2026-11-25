"""Input-view metrics, reported without claiming paper-equivalent benchmarks."""
import torch

from .losses import mesh_edges, temporal_acceleration


@torch.no_grad()
def deformation_metrics(vertices, rig, times):
    """Diagnostics on fitted frames; compare only runs sharing model/video/sampling."""
    edges, rest = mesh_edges(rig["vertices"], rig["faces"])
    if rest.numel():
        length = (vertices[:, edges[:, 0]]-vertices[:, edges[:, 1]]).norm(dim=-1)
        strain = (length-rest).abs()/rest.clamp_min(.25*rest.median())
        mean, p99 = float(strain.mean()), float(strain.quantile(.99, dim=1).mean())
    else:
        mean, p99 = 0., 0.
    acceleration = vertices[2:]-2*vertices[1:-1]+vertices[:-2]
    return {"edge_strain_mean": mean, "edge_strain_p99": p99,
            "mean_vertex_second_difference": float(acceleration.norm(dim=-1).mean())
            if len(vertices) > 2 else 0.,
            "vertex_acceleration_squared_per_second": float(temporal_acceleration(vertices, times))}


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
