"""Instance-level false-positive filter.

Every predicted instance gets a feature vector (size, shape, network
confidence, contrast against its surroundings, position on the disk). A
gradient-boosting classifier learns which instances match a ground-truth
filament (IoU > 0.5 for at least one annotator); instances scoring below a
threshold are dropped. It is trained on the held-out val split only, so the
network's predictions it sees are not overfit.
"""
from __future__ import annotations

import cv2
import numpy as np

FEATURES = [
    "log_area", "mean_prob", "max_prob", "p25_prob", "frac_conf", "extent", "elongation",
    "major_len", "contrast", "mean_gray", "radius", "n_inst", "rank_area", "group_mean",
]


def _disk(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))


def instance_features(inst: np.ndarray, prob: np.ndarray, gray_full: np.ndarray,
                      group: np.ndarray | None = None) -> np.ndarray:
    """(n_instances, len(FEATURES)) for labels 1..n of `inst` (at the map resolution)."""
    res = inst.shape[0]
    n = int(inst.max())
    if n == 0:
        return np.zeros((0, len(FEATURES)), np.float32)
    gray = cv2.resize(gray_full, (res, res), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    areas = np.bincount(inst.ravel(), minlength=n + 1)[1:].astype(np.float32)
    order = np.argsort(np.argsort(-areas))  # 0 = largest
    ring_k = _disk(max(2, res // 128))
    rows = []
    for k in range(1, n + 1):
        m = inst == k
        ys, xs = np.nonzero(m)
        p = prob[m]
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        extent = len(ys) / float((y1 - y0) * (x1 - x0))
        if len(ys) > 2:
            ev = np.sort(np.linalg.eigvalsh(np.cov(np.stack([xs, ys]).astype(np.float64))))
            elong = float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
            major = float(4 * np.sqrt(max(ev[1], 0)))
        else:
            elong, major = 1.0, 1.0
        # contrast: instance mean vs a ring around it that excludes all predicted foreground
        sub = (slice(max(0, y0 - 16), y1 + 16), slice(max(0, x0 - 16), x1 + 16))
        ring = cv2.dilate(m[sub].astype(np.uint8), ring_k).astype(bool) & (inst[sub] == 0)
        g_in = float(gray[m].mean())
        g_out = float(gray[sub][ring].mean()) if ring.any() else g_in
        cy, cx = ys.mean(), xs.mean()
        rows.append([
            np.log(len(ys)), float(p.mean()), float(p.max()), float(np.percentile(p, 25)),
            float((p > 0.8).mean()), extent, elong, major / res,
            g_in / max(g_out, 1e-3), g_in, float(np.hypot(cx - res / 2, cy - res / 2) / (res / 2)),
            float(n), float(order[k - 1]),
            float(group[m].mean()) if group is not None else -1.0,
        ])
    return np.asarray(rows, np.float32)
