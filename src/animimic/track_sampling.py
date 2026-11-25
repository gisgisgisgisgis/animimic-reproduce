"""Distribute track queries across visible rig parts."""
import numpy as np


def track_part_ids(rig, triangles, barycentric, groups):
    """Assign each surface point to the part with most interpolated skin weight."""
    names = ["other", *sorted(groups)]
    if "other" in groups or any(not joints for joints in groups.values()):
        raise ValueError("Track groups must be nonempty and cannot be named 'other'")
    weights = rig["weights"]
    membership = np.zeros(weights.shape[1], dtype=int)
    for part, name in enumerate(names[1:], 1):
        joints = np.asarray(groups[name])
        if (joints.dtype.kind not in "iu" or np.any((joints < 0) | (joints >= len(membership)))
                or len(np.unique(joints)) != len(joints) or membership[joints].any()):
            raise ValueError("Track groups require valid, nonoverlapping joint indices")
        membership[joints] = part
    face_weights = weights[rig["faces"][triangles]]
    point_weights = (face_weights * barycentric[:, :, None]).sum(axis=1)
    part_weights = point_weights @ (membership[:, None] == np.arange(len(names)))
    return names, part_weights.argmax(axis=1)


def spread_samples(pixels, parts, count):
    """Divide the budget among visible parts and spread points within each part."""
    if count < 1:
        raise ValueError("supervision.max_tracks must be positive")
    groups = [np.flatnonzero(parts == part) for part in np.unique(parts)]
    quotas = np.zeros(len(groups), dtype=int)
    for _ in range(min(count, len(pixels))):
        group = min((i for i, ids in enumerate(groups) if quotas[i] < len(ids)),
                    key=lambda i: (quotas[i], i))
        quotas[group] += 1
    selected = []
    for ids, quota in zip(groups, quotas):
        xy = pixels[ids].astype(np.float32)
        distances = ((xy - xy.mean(axis=0))**2).sum(axis=1)
        next_point = distances.argmin()
        nearest = np.full(len(ids), np.inf)
        for _ in range(quota):
            selected.append(ids[next_point])
            nearest = np.minimum(nearest, ((xy - xy[next_point])**2).sum(axis=1))
            nearest[next_point] = -1
            next_point = nearest.argmax()
    return np.asarray(selected, dtype=int)
