"""Classical (non-learned) filament segmentation baseline.

Pipeline, run at a reduced working resolution for speed:
  1. Locate the solar disk and fit a circle from the bright-region moments.
  2. Flatten limb darkening with a radial median profile, then remaining
     large-scale brightness with a heavy median-filtered background.
  3. Hysteresis-threshold dark pixels relative to that background: strong
     seeds (ratio < thresh_high) grow into weaker connected pixels
     (ratio < thresh_low), restricted to the inner disk (the limb is noisy).
  4. Clean up with morphology, drop small blobs and sunspot-like blobs
     (very dark, compact cores).
  5. Group nearby fragments into one filament by labelling a dilated copy of
     the mask, then upsample the instance labels to 2048x2048.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class Params:
    work_size: int = 1024        # processing resolution
    bg_kernel: int = 61          # median background kernel (work px, odd)
    smooth_sigma: float = 1.5    # gaussian denoise before thresholding
    thresh_high: float = 0.85    # seed if pixel / background < thresh_high
    thresh_low: float = 0.93     # grow seeds into pixels below thresh_low
    disk_margin: float = 0.95    # keep r < margin * R
    open_radius: int = 1
    close_radius: int = 2
    min_area: int = 60           # work px, after cleaning
    group_radius: int = 6        # fragments within ~2*radius are one filament
    sunspot_thresh: float = 0.55 # components whose min ratio is below this ...
    sunspot_max_area: int = 1500 # ... and are small are treated as sunspots
    out_dilate: int = 0          # grow each instance (work px); annotations are drawn generously

    def to_dict(self) -> dict:
        return asdict(self)


def _disk(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))


def radial_profile(img: np.ndarray, rr: np.ndarray, r: float, in_disk: np.ndarray,
                   n_bins: int = 200) -> np.ndarray:
    """Median brightness as a function of radius, broadcast back to the image."""
    b = np.clip((rr / r * n_bins).astype(np.int32), 0, n_bins)
    prof = np.ones(n_bins + 1, np.float32)
    vals, bins = img[in_disk], b[in_disk]
    order = np.argsort(bins, kind="stable")
    vals, bins = vals[order], bins[order]
    starts = np.searchsorted(bins, np.arange(n_bins + 1))
    ends = np.searchsorted(bins, np.arange(n_bins + 1), side="right")
    for i in range(n_bins + 1):
        if ends[i] > starts[i]:
            prof[i] = np.median(vals[starts[i]:ends[i]])
    return prof[b]


def hysteresis(weak: np.ndarray, strong: np.ndarray) -> np.ndarray:
    """Keep connected components of `weak` that contain at least one `strong` pixel."""
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[strong & weak])] = True
    keep[0] = False
    return keep[lab]


def fit_disk(gray: np.ndarray) -> tuple[float, float, float]:
    """Return (cx, cy, r) of the solar disk."""
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = (gray > thr).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(bw, 8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (lab == k).astype(np.uint8)
    # fill holes (dark filaments / sunspots inside the disk)
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (cx, cy), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    area = cv2.contourArea(max(cnts, key=cv2.contourArea))
    r_area = np.sqrt(area / np.pi)
    return float(cx), float(cy), float(min(r, r_area))


def segment(gray: np.ndarray, p: Params) -> np.ndarray:
    """Return an int32 instance label image at the input resolution."""
    h0, w0 = gray.shape
    s = p.work_size / w0
    g = cv2.resize(gray, (p.work_size, int(round(h0 * s))), interpolation=cv2.INTER_AREA)
    cx, cy, r = fit_disk(g)

    yy, xx = np.mgrid[: g.shape[0], : g.shape[1]]
    rr = np.hypot(xx - cx, yy - cy)
    in_disk = rr < r
    inner = rr < p.disk_margin * r

    gf = g.astype(np.float32)
    flat = gf / np.maximum(radial_profile(gf, rr, r, in_disk), 1.0)
    # Fill outside-disk with 1 so the background filter isn't pulled down at the limb.
    flat = np.where(in_disk, flat, 1.0)
    flat8 = np.clip(flat * 128.0, 0, 255).astype(np.uint8)
    bg = cv2.medianBlur(flat8, p.bg_kernel).astype(np.float32) / 128.0
    raw_ratio = flat / np.maximum(bg, 1e-3)
    ratio = cv2.GaussianBlur(raw_ratio, (0, 0), p.smooth_sigma) if p.smooth_sigma > 0 else raw_ratio

    dark = hysteresis(ratio < p.thresh_low, (ratio < p.thresh_high) & inner) & inner
    dark = dark.astype(np.uint8)
    if p.open_radius > 0:
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, _disk(p.open_radius))
    if p.close_radius > 0:
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, _disk(p.close_radius))
    dark &= inner.astype(np.uint8)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    keep = np.zeros(n, bool)
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < p.min_area:
            continue
        if area <= p.sunspot_max_area and raw_ratio[lab == k].min() < p.sunspot_thresh:
            continue
        keep[k] = True
    dark = keep[lab].astype(np.uint8)

    # Group fragments: components of the dilated mask define instances.
    grouped = cv2.dilate(dark, _disk(p.group_radius)) if p.group_radius > 0 else dark
    _, glab = cv2.connectedComponents(grouped, connectivity=8)
    inst = np.where(dark > 0, glab, 0).astype(np.int32)
    if p.out_dilate > 0:
        # grey dilation grows labels into background; where labels meet the larger id wins
        grown = cv2.dilate(inst.astype(np.float32), _disk(p.out_dilate))
        inst = np.where(inst > 0, inst, grown).astype(np.int32)

    # Relabel compactly, then upsample with nearest neighbour.
    ids = np.unique(inst)
    ids = ids[ids > 0]
    lut = np.zeros(int(inst.max()) + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    inst = lut[inst]
    return cv2.resize(inst.astype(np.float32), (w0, h0), interpolation=cv2.INTER_NEAREST).astype(np.int32)
