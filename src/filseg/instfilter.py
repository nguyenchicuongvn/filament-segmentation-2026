"""Instance-level false-positive filter.

Every predicted instance gets a feature vector (size, shape, network
confidence, contrast against its surroundings, position on the disk). A
gradient-boosting model scores each instance and those below a threshold are
dropped. Two kinds (scripts/fp_filter.py --target):
  clf  classifier: does the instance match a filament (IoU > 0.5) for at
       least one annotator?
  iou  regressor: the instance's IoU with its match averaged over the image's
       annotators (0 where unmatched) -- its expected contribution to PQ's
       numerator. With pixel-disjoint masks PQ is additive over instances, and
       keeping an instance raises PQ iff its expected IoU exceeds about PQ/2.
It is trained on held-out (val or out-of-fold) maps only, so the network's
predictions it sees are not overfit.

FEATURES grows over time; a saved filter records the names it was trained on
and only those columns are passed to it.

AGREE_FEATURES (optional, appended after FEATURES when used) measure how far
four other holdout models agree with the base model on each instance; they need
those models' maps at inference, so they are kept out of FEATURES.
"""
from __future__ import annotations

import cv2
import numpy as np

FEATURES = [
    "log_area", "mean_prob", "max_prob", "p25_prob", "frac_conf", "extent", "elongation",
    "major_len", "contrast", "mean_gray", "radius", "n_inst", "rank_area", "group_mean",
    # position on the solar disk: annotators skip filaments beyond ~70 deg from the central meridian
    "abs_x_disk", "y_disk", "abs_lon",
]


# cross-model agreement; the other models in a fixed order (r34, r34_grp, r34_e60, cnxt)
AGREE_FEATURES = [
    "agree_mean_m1", "agree_mean_m2", "agree_mean_m3", "agree_mean_m4",
    "agree_std5", "agree_min4", "agree_vote3", "agree_iou4",
]
ALL_FEATURES = FEATURES + AGREE_FEATURES


def _disk(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))


def _disk_position(cx: float, cy: float, dcx: float, dcy: float, dr: float) -> tuple[float, float, float]:
    """|x|, y on the disk (radius 1) and |longitude| in degrees from the central meridian."""
    x, y = (cx - dcx) / dr, (cy - dcy) / dr
    lon = np.degrees(np.arcsin(np.clip(abs(x) / max(np.sqrt(max(1 - y * y, 1e-6)), 1e-6), 0, 1)))
    return abs(x), y, float(lon)


def select(feats: np.ndarray, names: list[str]) -> np.ndarray:
    """The columns of a filter trained on `names` (older filters use fewer features).
    `feats` has the FEATURES columns, optionally followed by the AGREE_FEATURES ones."""
    return feats[:, [ALL_FEATURES.index(n) for n in names]]


def score(filt: dict, feats: np.ndarray) -> np.ndarray:
    """Keep-score of each instance for a saved filter (probability or expected IoU)."""
    X = select(feats, filt["features"])
    if filt.get("kind", "clf") == "iou":
        return filt["model"].predict(X)
    return filt["model"].predict_proba(X)[:, 1]


def instance_features(inst: np.ndarray, prob: np.ndarray, gray_full: np.ndarray,
                      group: np.ndarray | None = None) -> np.ndarray:
    """(n_instances, len(FEATURES)) for labels 1..n of `inst` (at the map resolution)."""
    res = inst.shape[0]
    n = int(inst.max())
    if n == 0:
        return np.zeros((0, len(FEATURES)), np.float32)
    from .classical import fit_disk

    gray8 = cv2.resize(gray_full, (res, res), interpolation=cv2.INTER_AREA)
    dcx, dcy, dr = fit_disk(gray8)
    gray = gray8.astype(np.float32) / 255.0
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
            *_disk_position(cx, cy, dcx, dcy, dr),
        ])
    return np.asarray(rows, np.float32)


def _components(prob: np.ndarray, thresh: float = 0.5, min_area: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """8-connected components of prob > thresh with area >= min_area (others set to 0), and areas."""
    _, lab, stats, _ = cv2.connectedComponentsWithStats((prob > thresh).astype(np.uint8), connectivity=8)
    drop = stats[:, cv2.CC_STAT_AREA] < min_area
    drop[0] = True
    return np.where(drop[lab], 0, lab), stats[:, cv2.CC_STAT_AREA].astype(np.float64)


def agreement_features(inst: np.ndarray, base_prob: np.ndarray, other_probs: list[np.ndarray]) -> np.ndarray:
    """(n_instances, len(AGREE_FEATURES)) for labels 1..n of `inst`, made from `base_prob` (its
    model's map) and the 4 other models' maps, all at the same resolution:
      agree_mean_mj  mean of model j's prob inside the instance
      agree_std5     std of the 5 per-model instance means (base model included)
      agree_min4     min of the 4 other means
      agree_vote3    fraction of instance pixels where >= 3 of the 5 models have p > 0.5
      agree_iou4     mean over the 4 other models of the best IoU between the instance and a
                     component (8-connected, area >= 50) of that model's p > 0.5 map
    """
    if len(other_probs) != 4:
        raise ValueError(f"agreement_features needs 4 other models, got {len(other_probs)}")
    n = int(inst.max())
    if n == 0:
        return np.zeros((0, len(AGREE_FEATURES)), np.float32)
    flat = inst.ravel()
    area = np.bincount(flat, minlength=n + 1).astype(np.float64)[1:]
    probs = [base_prob, *other_probs]
    means = np.stack([np.bincount(flat, weights=p.ravel().astype(np.float64), minlength=n + 1)[1:] / area
                      for p in probs], 1)
    votes = sum((p > 0.5).astype(np.int32) for p in probs)
    vote3 = np.bincount(flat, weights=(votes.ravel() >= 3).astype(np.float64), minlength=n + 1)[1:] / area
    best_iou = np.zeros((n, len(other_probs)))
    for j, p in enumerate(other_probs):
        lab, carea = _components(p)
        lab = lab.ravel()
        both = (flat > 0) & (lab > 0)
        if not both.any():
            continue
        L = len(carea)
        pairs, inter = np.unique(flat[both].astype(np.int64) * L + lab[both], return_counts=True)
        ki, ci = pairs // L, pairs % L
        iou = inter / (area[ki - 1] + carea[ci] - inter)
        np.maximum.at(best_iou[:, j], ki - 1, iou)
    others = means[:, 1:]
    return np.column_stack([
        others, means.std(1), others.min(1), vote3, best_iou.mean(1),
    ]).astype(np.float32)
