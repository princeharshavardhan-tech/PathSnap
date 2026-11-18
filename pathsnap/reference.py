"""Reference data is accessible only to offline training and post-prediction scoring."""
import numpy as np
from .data import project, read_table, GAP


def align(trip, reference_path):
    route = read_table(reference_path, 1, np.int32)[:, 0]
    if not len(route) or not np.isin(route, trip.edge_ids).all():
        raise ValueError("Reference route contains missing or unusable arc IDs")
    edges = trip.edges[route]
    a = trip.xy[edges[:, 0]]
    vec = trip.xy[edges[:, 1]] - a
    cols = np.arange(len(route), dtype=np.int32)
    prev = np.zeros(len(route))
    pointer_dtype = np.uint16 if len(route) <= 65535 else np.uint32
    parent = np.empty((len(trip.gps), len(route)), dtype=pointer_dtype)
    for begin in range(0, len(trip.gps), 64):
        distances, _ = project(trip.gps[begin:begin+64, None], a, vec)
        for j, costs in enumerate(np.log1p(distances)):
            prefix = np.minimum.accumulate(prev)
            improvements = np.r_[True, prev[1:] < prefix[:-1]]
            parent[begin+j] = np.maximum.accumulate(np.where(improvements, cols, 0))
            prev = costs + prefix
    pos = np.empty(len(trip.gps), dtype=np.int32)
    pos[-1] = np.argmin(prev)
    for i in range(len(pos)-1, 0, -1):
        pos[i-1] = parent[i, pos[i]]
    dist, frac = project(trip.gps, a[pos], vec[pos])
    lengths = np.linalg.norm(vec, axis=1)
    along = np.r_[0., np.cumsum(lengths)[:-1]][pos] + frac * lengths[pos]
    raw_step = np.linalg.norm(np.diff(trip.gps, axis=0), axis=1)
    ok = (np.diff(trip.track[:, 2]) > 0) & (np.diff(trip.track[:, 2]) <= GAP)
    residual = np.abs(np.maximum(0, np.diff(along))-raw_step)[ok]
    return route, route[pos], dist, residual
