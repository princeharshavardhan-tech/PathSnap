"""Read trip inputs and build candidate features without access to a reference route."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

K = 8
GAP = 30.0
ANCHOR_SPACING = 35.0
FEATURES = ["log_distance", "log_distance_excess", "heading_cos", "incoming_cos",
            "outgoing_cos", "log_speed", "log_arc_length", "log_endpoint_margin", "fraction"]


def read_table(path, columns, dtype=float):
    values = np.loadtxt(path, ndmin=2, dtype=dtype)
    if values.shape[1] != columns:
        raise ValueError(f"{path.name} needs {columns} columns, found {values.shape[1]}")
    return values


def valid_lonlat(a):
    return np.isfinite(a).all(axis=1) & (np.abs(a[:, 0]) <= 180) & (np.abs(a[:, 1]) <= 90)


@dataclass
class Trip:
    track: np.ndarray
    nodes: np.ndarray
    edges: np.ndarray
    xy: np.ndarray
    gps: np.ndarray
    edge_ids: np.ndarray
    source_rows: np.ndarray
    raw_count: int
    missing: int


def read_inputs(folder):
    folder = Path(folder)
    base = folder / folder.name
    raw = read_table(base.with_suffix(".track"), 3)
    nodes = read_table(base.with_suffix(".nodes"), 2)
    edges = read_table(base.with_suffix(".arcs"), 2, np.int32)
    valid = valid_lonlat(raw[:, :2]) & np.isfinite(raw[:, 2])
    order = np.flatnonzero(valid)
    order = order[np.argsort(raw[order, 2], kind="stable")]
    track = raw[order]
    if not len(track):
        raise ValueError("No valid observations")
    good = (edges >= 0).all(axis=1) & (edges < len(nodes)).all(axis=1)
    idx = np.flatnonzero(good)
    idx = idx[valid_lonlat(nodes)[edges[idx]].all(axis=1)]
    if not len(idx):
        raise ValueError("No usable road segments")
    # A local tangent-plane approximation, centred on the trip.
    origin = np.median(track[:, :2], axis=0)
    scale = 6371008.8 * np.pi / 180 * np.array([np.cos(np.deg2rad(origin[1])), 1.])
    xy = (nodes - origin) * scale
    gps = (track[:, :2] - origin) * scale
    delta = xy[edges[idx, 1]] - xy[edges[idx, 0]]
    idx = idx[np.linalg.norm(delta, axis=1) > 1e-5]
    if not len(idx):
        raise ValueError("All road segments have zero length")
    return Trip(track, nodes, edges, xy, gps, idx.astype(np.int32), order, len(raw),
                int((~np.isfinite(raw)).sum()))


def project(points, start, direction):
    length2 = np.einsum("...i,...i->...", direction, direction)
    f = np.clip(np.einsum("...i,...i->...", points - start, direction) /
                np.maximum(length2, 1e-15), 0, 1)
    return np.linalg.norm(points - start - f[..., None] * direction, axis=-1), f


class SegmentIndex:
    """Exact nearest directed segments using a KD-tree of regularly spaced anchors."""

    def __init__(self, trip):
        self.ids = trip.edge_ids
        e = trip.edges[self.ids]
        self.start = trip.xy[e[:, 0]]
        self.delta = trip.xy[e[:, 1]] - self.start
        self.length = np.linalg.norm(self.delta, axis=1)
        parts = np.maximum(1, np.ceil(self.length / ANCHOR_SPACING).astype(int))
        self.owners = np.repeat(np.arange(len(parts), dtype=np.int32), parts)
        first = np.cumsum(parts) - parts
        fraction = (np.arange(parts.sum()) - first[self.owners] + .5) / parts[self.owners]
        self.tree = cKDTree(self.start[self.owners] + self.delta[self.owners] * fraction[:, None])

    def nearest(self, points, k=K):
        k = min(k, len(self.ids))
        choices = np.empty((len(points), k), np.int32)
        for start in range(0, len(points), 512):
            pts = points[start:start + 512]
            remaining = np.arange(len(pts))
            n = min(max(4*k, 24), len(self.owners))
            while len(remaining):
                a_dist, a_idx = self.tree.query(pts[remaining], k=list(range(1, n + 1)))
                ids = np.sort(self.owners[a_idx], axis=1)
                d, _ = project(pts[remaining, None], self.start[ids], self.delta[ids])
                d[:, 1:] = np.where(ids[:, 1:] == ids[:, :-1], np.inf, d[:, 1:])
                order = np.argsort(np.round(d, 7), axis=1, kind="stable")[:, :k]
                best = np.take_along_axis(ids, order, axis=1)
                far = np.take_along_axis(d, order[:, -1:], axis=1)[:, 0]
                # Every point of a segment is within half an anchor spacing of one anchor.
                certified = (far + ANCHOR_SPACING/2 + 1e-6 < a_dist[:, -1]) | (n == len(self.owners))
                choices[start + remaining[certified]] = best[certified]
                remaining = remaining[~certified]
                n = min(n * 2, len(self.owners))
        return self.ids[choices]


def motion(trip):
    t, xy = trip.track[:, 2], trip.gps
    ix = np.arange(len(t))
    boundaries = np.r_[0, np.flatnonzero(np.diff(t) > GAP) + 1, len(t)]
    left = np.empty_like(ix)
    right = np.empty_like(ix)
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        left[a:b] = np.maximum(ix[a:b]-4, a)
        right[a:b] = np.minimum(ix[a:b]+4, b-1)
    before, after = xy - xy[left], xy[right] - xy
    center = before + after
    vectors = [v / np.maximum(np.linalg.norm(v, axis=1)[:, None], 2.0) for v in (center, before, after)]
    speed = np.linalg.norm(center, axis=1) / np.maximum(t[right]-t[left], 1.)
    return vectors, speed


def candidates(trip):
    ids = SegmentIndex(trip).nearest(trip.gps)
    edges = trip.edges[ids]
    a = trip.xy[edges[:, :, 0]]
    vec = trip.xy[edges[:, :, 1]] - a
    dist, frac = project(trip.gps[:, None], a, vec)
    length = np.linalg.norm(vec, axis=2)
    unit = vec / length[:, :, None]
    headings, speed = motion(trip)
    cosines = [np.einsum("td,tkd->tk", h, unit) for h in headings]
    features = [np.log1p(dist), np.log1p(np.maximum(0, dist-dist.min(axis=1, keepdims=True))),
                *cosines, np.broadcast_to(np.log1p(np.minimum(speed, 80))[:, None], dist.shape),
                np.log1p(length), np.log1p(length*np.minimum(frac, 1-frac)), frac]
    return ids, np.stack(features, axis=-1).astype(np.float32)
