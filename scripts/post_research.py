"""Cheap post-processing variants on existing probability maps, cross-tuned against hyst_soft.

Research card 05 (no retraining). Baseline = compare.py's tune:hyst_soft grid (scaled to the map
resolution), reproduced here with a faster exact IoU (bincount over GT pixels instead of RLE).

Two-stage cross-tuning, honest: in each split (the 3 random halves x 2 directions of compare.py)
the hyst_soft grid is tuned on half A; a variant then tunes its own extra grid on half A given that
choice, and half B is scored. Every variant grid contains the identity, so the variant can only
differ from the baseline by what half A chose. Paired bootstrap by image (seed 0, 2000 resamples),
the same as compare.py.

Variants (areas / lengths in px at 2048; scaled with the map resolution):
  radius   group_radius of the fragment grouping (baseline 12 px at 2048)
  orient   grouping by an oriented dilation: each component is dilated by a line of half-length L
           along its major axis (union a disk of radius r), so fragments of one elongated filament
           merge along it but not sideways
  seedgrp  umerkang S11c style: instances are groups of seed components (p > seed_thresh) whose
           gap is < d; every extent pixel joins the nearest seed group (watershed on -p inside
           the extent), extent components without a seed dropped
  small    shrink instances smaller than small_area to p > small_thresh
  large    grow instances >= large_area by up to `grow` px into unclaimed p > large_thresh
  peakrel  per-instance threshold max(t_lo, a * peak): shrink to it and grow (up to `grow` px) into it
  smooth   Gaussian blur of the map (sigma) before the pipeline, and/or fill holes per instance

Usage:
  python scripts/post_research.py outputs/ens/n1n2_f0 --variants radius,orient,small --jobs 6
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.data import load_train, with_maps  # noqa: E402
from filseg.dl import fill_instance_holes  # noqa: E402
from filseg.metric import IOU_THRESHOLD  # noqa: E402

cv2.setNumThreads(1)  # one thread per worker (shared machine; --jobs sets the parallelism)

# compare.py GRIDS["hyst_soft"] at 2048 px (min_area 50/90 and radius 6 at 1024)
BASE = {"thresh": [0.2, 0.25, 0.3, 0.4, 0.5], "seed_thresh": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        "min_area": [200, 360], "group_radius": [12]}

def _grid(identity: dict, **grid) -> list[dict]:
    """The identity (baseline-equivalent) combo first, then the product of `grid`."""
    return [identity] + [dict(zip(grid, v)) for v in itertools.product(*grid.values())]


VARIANTS = {
    "radius": [{"r": r} for r in (12, 0, 6, 18, 24, 32, 44)],
    "orient": [{"L": 0, "r": 12}] + [{"L": L, "r": r} for L in (16, 32, 64, 96) for r in (12, 6)],
    "seedgrp": [{"d": d} for d in (None, 8, 16, 32, 48)],
    "small": _grid({"small_area": 0}, small_area=[800, 1600, 3200], small_thresh=[0.4, 0.5, 0.6]),
    "large": _grid({"large_area": 0}, large_area=[3200, 8000, 20000], large_thresh=[0.1, 0.2], grow=[6, 16]),
    "peakrel": _grid({"a": 0.0}, a=[0.3, 0.5, 0.7], t_lo=[0.1, 0.2], grow=[0, 8]),
    "smooth": [{"sigma": sg, "fill": f} for sg in (0, 1, 2, 4) for f in (False, True)],
    # post hoc (after orient passed on the OOF with L16 r6 in every split): ablation of the line
    # (L 0 vs 16) and of the radius (12 vs 6), and a finer grid around L16 r6
    "orient_ab": [{"L": 0, "r": 12}, {"L": 0, "r": 6}, {"L": 16, "r": 12}, {"L": 16, "r": 6}],
    "orient_fine": [{"L": 0, "r": 12}] + [{"L": L, "r": r} for L in (8, 16, 24) for r in (4, 6, 8)],
    # detection (components, seeds) on the ensemble map; extent re-drawn on q = alpha * N2 +
    # (1 - alpha) * N1 (the ensemble is alpha = 0.5) at threshold base thresh + delta, within
    # `grow` px of the instance. Needs --members N1_dirs:N2_dirs.
    "extmix": _grid({"alpha": None}, alpha=[0.0, 0.25, 0.75, 1.0], delta=[-0.1, 0.0, 0.1], grow=[8]),
}
SCALED = {"min_area", "small_area", "large_area"}           # areas: scale with res^2
LENGTHS = {"group_radius", "r", "L", "d", "grow", "sigma"}  # lengths: scale with res


def _disk(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))


def _scale(d: dict, s: float) -> dict:
    out = {}
    for k, v in d.items():
        if v is None or isinstance(v, bool):
            out[k] = v
        elif k in SCALED:
            out[k] = round(v * s * s)
        elif k in LENGTHS:
            out[k] = v * s if k == "sigma" else round(v * s)
        else:
            out[k] = v
    return out


def combos(grid: dict) -> list[dict]:
    return [dict(zip(grid, v)) for v in itertools.product(*grid.values())]


# ---------------------------------------------------------------- instance construction

def kept_components(prob, t, seed, min_area):
    """Components of p > t (8-conn) with area >= min_area and peak > seed: (fg uint8, lab, keep)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((prob > t).astype(np.uint8), connectivity=8)
    keep = st[:, cv2.CC_STAT_AREA] >= min_area
    if n > 1:
        keep &= _peaks(prob, lab, n) > seed
    keep[0] = False
    return keep[lab].astype(np.uint8), lab, keep, st


def _peaks(prob, lab, n):
    """Max of prob per label 0..n-1 (fast: only labelled pixels)."""
    m = lab > 0
    peak = np.zeros(n, np.float32)
    np.maximum.at(peak, lab[m], prob[m])
    return peak


def _bbox(mask, pad):
    ys = np.flatnonzero(mask.any(1))
    xs = np.flatnonzero(mask.any(0))
    if not len(ys):
        return None
    H_, W_ = mask.shape
    return (slice(max(ys[0] - pad, 0), min(ys[-1] + 1 + pad, H_)),
            slice(max(xs[0] - pad, 0), min(xs[-1] + 1 + pad, W_)))


def group_disk(fg, r):
    """CC of fg dilated by disk(r), restricted to fg (computed on fg's bounding box, exact)."""
    out = np.zeros(fg.shape, np.int32)
    sl = _bbox(fg, r + 1)
    if sl is None:
        return out
    f = fg[sl]
    grouped = cv2.dilate(f, _disk(r)) if r > 0 else f
    _, glab = cv2.connectedComponents(grouped, connectivity=8)
    out[sl] = np.where(f > 0, glab, 0)
    return out


def group_oriented(fg, lab, keep, st, L, r):
    """Dilate each kept component by disk(r) union a line of half-length L along its major axis."""
    if L <= 0:
        return group_disk(fg, r)
    Hh, Ww = fg.shape
    grouped = np.zeros_like(fg)
    pad = L + r + 1
    for k in np.nonzero(keep)[0]:
        x, y, w, h = st[k, :4]
        x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + w + pad, Ww), min(y + h + pad, Hh)
        m = (lab[y0:y1, x0:x1] == k).astype(np.uint8)
        mo = cv2.moments(m, binaryImage=True)
        th = 0.5 * np.arctan2(2 * mo["mu11"], mo["mu20"] - mo["mu02"])
        ker = np.zeros((2 * L + 1, 2 * L + 1), np.uint8)
        dx, dy = L * np.cos(th), L * np.sin(th)
        cv2.line(ker, (round(L - dx), round(L - dy)), (round(L + dx), round(L + dy)), 1, 3)
        if r > 0:
            ker[L - r:L + r + 1, L - r:L + r + 1] |= _disk(r)
        grouped[y0:y1, x0:x1] |= cv2.dilate(m, ker)
    _, glab = cv2.connectedComponents(grouped, connectivity=8)
    return np.where(fg > 0, glab, 0).astype(np.int32)


def group_seeds(prob, fg, seed, d):
    """Seed components (p > seed inside fg) grouped when closer than d; extent pixels to the
    nearest seed group by watershed on -p within fg."""
    from skimage.segmentation import watershed
    seeds = ((prob > seed) & (fg > 0)).astype(np.uint8)
    grouped = cv2.dilate(seeds, _disk(max(d // 2, 1))) if d > 0 else seeds
    _, glab = cv2.connectedComponents(grouped, connectivity=8)
    markers = np.where(seeds > 0, glab, 0).astype(np.int32)
    ys, xs = np.nonzero(fg)
    if not len(ys):
        return np.zeros(fg.shape, np.int32)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    out = np.zeros(fg.shape, np.int32)
    out[y0:y1, x0:x1] = watershed(-prob[y0:y1, x0:x1], markers[y0:y1, x0:x1], mask=fg[y0:y1, x0:x1] > 0)
    return out


def grow_into(inst, allowed, steps):
    """Grow labels up to `steps` px (3x3 steps) into unclaimed pixels where allowed[label] holds
    (allowed: bool image per pixel computed for the candidate label via a callable)."""
    out = inst.copy()
    k = np.ones((3, 3), np.uint8)
    for _ in range(steps):
        grown = cv2.dilate(out.astype(np.float32), k).astype(np.int32)
        add = (out == 0) & (grown > 0) & allowed(grown)
        if not add.any():
            break
        out[add] = grown[add]
    return out


def relabel(inst, min_area):
    nz = np.flatnonzero(inst)
    out = np.zeros(inst.shape, np.int32)
    if not len(nz):
        return out
    v = inst.ravel()[nz]
    areas = np.bincount(v)
    keep = areas >= min_area
    keep[0] = False
    lut = np.zeros(len(areas), np.int32)
    lut[keep] = np.arange(1, int(keep.sum()) + 1)
    out.ravel()[nz] = lut[v]
    return out


def _bbox_apply(inst, fn, pad):
    """Run fn on the bounding box of the instances (+pad) -- the maps are ~99.7% background."""
    ys, xs = np.nonzero(inst)
    if not len(ys):
        return inst
    H_, W_ = inst.shape
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + 1 + pad, H_)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + 1 + pad, W_)
    out = inst.copy()
    out[y0:y1, x0:x1] = fn(inst[y0:y1, x0:x1], (slice(y0, y1), slice(x0, x1)))
    return out


def build(prob, b: dict, v: str | None = None, e: dict | None = None, members=None) -> np.ndarray:
    """Instance labels for base params b, optionally with variant v and its extra params e."""
    e = e or {}
    if v == "smooth" and e.get("sigma"):
        prob = cv2.GaussianBlur(prob, (0, 0), e["sigma"])
    t, seed, ma, r = b["thresh"], b["seed_thresh"], b["min_area"], b["group_radius"]
    fg, lab, keep, st = kept_components(prob, t, seed, ma)
    if v == "radius":
        inst = group_disk(fg, e["r"])
    elif v and v.startswith("orient"):
        inst = group_oriented(fg, lab, keep, st, e["L"], e["r"])
    elif v == "seedgrp" and e["d"] is not None:
        inst = group_seeds(prob, fg, seed, e["d"])
    else:
        inst = group_disk(fg, r)

    if v == "smooth" and e.get("fill"):
        inst = fill_instance_holes(inst)
    elif v == "small" and e["small_area"]:
        areas = np.bincount(inst.ravel())
        small = np.zeros(len(areas), bool)
        small[1:] = (areas[1:] > 0) & (areas[1:] < e["small_area"])
        inst = np.where(small[inst] & (prob <= e["small_thresh"]), 0, inst)
    elif v == "large" and e["large_area"]:
        areas = np.bincount(inst.ravel())
        large = np.zeros(len(areas), bool)
        large[1:] = areas[1:] >= e["large_area"]
        lt, g = e["large_thresh"], e["grow"]

        def fn(sub, sl):
            p = prob[sl]
            seedl = np.where(large[sub], sub, 0)
            grown = grow_into(seedl, lambda gr: p > lt, g)
            return np.where((sub == 0) & (grown > 0), grown, sub)
        inst = _bbox_apply(inst, fn, g + 1)
    elif v == "peakrel" and e["a"]:
        n = int(inst.max()) + 1
        pk = _peaks(prob, inst, n)
        thr = np.maximum(e["t_lo"], e["a"] * pk)
        thr[0] = 2.0
        inst = np.where(prob > thr[inst], inst, 0)
        if e["grow"]:
            def fn(sub, sl):
                p = prob[sl]
                return grow_into(sub, lambda gr: p > thr[gr], e["grow"])
            inst = _bbox_apply(inst, fn, e["grow"] + 1)
    elif v == "extmix" and e["alpha"] is not None:
        n1, n2 = members
        q = e["alpha"] * n2 + (1 - e["alpha"]) * n1
        te, g = t + e["delta"], e["grow"]

        def fn(sub, sl):
            qq = q[sl]
            sub = np.where(qq > te, sub, 0)
            return grow_into(sub, lambda gr: qq > te, g)
        inst = _bbox_apply(inst, fn, g + 1)
    return relabel(inst, ma)


# ---------------------------------------------------------------- scoring

def stats(inst: np.ndarray, gts: list[list[np.ndarray]]) -> np.ndarray:
    """(sum TP IoU, TP, FP, FN) of one image over its annotator entries; gts[a] = flat pixel
    indices of each GT mask of annotator a. Exact IoU (same as pycocotools on the masks)."""
    flat = inst.ravel()
    n = int(flat.max())
    area_p = np.bincount(flat[np.flatnonzero(flat)], minlength=n + 1)[1:]
    out = np.zeros(4)
    for gt in gts:
        if not gt:
            out[2] += n
            continue
        if not n:
            out[3] += len(gt)
            continue
        inter = np.stack([np.bincount(flat[idx], minlength=n + 1)[1:] for idx in gt])
        area_g = np.array([len(idx) for idx in gt])[:, None]
        iou = inter / (area_g + area_p[None] - inter)
        hit = iou > IOU_THRESHOLD
        out += [iou[hit].sum(), hit.sum(), (hit.sum(0) == 0).sum(), (hit.sum(1) == 0).sum()]
    return out


def _gts(rles_by_ann):
    out = []
    for rles in rles_by_ann:
        if not rles:
            out.append([])
            continue
        m = mask_util.decode(rles)  # (H, W, n)
        out.append([np.flatnonzero(m[:, :, i]) for i in range(m.shape[2])])
    return out


def _load(probs, stem):
    return np.load(Path(probs) / f"{stem}.npy").astype(np.float32) / 255.0


def _job_base(args):
    """build(prob, b) for every base combo, sharing the components per threshold and the stats of
    combos that keep the same components (exactly the same result as calling build per combo)."""
    probs, stem, rles, base = args
    prob, gts = _load(probs, stem), _gts(rles)
    cc, memo, rows = {}, {}, []
    for b in base:
        t = b["thresh"]
        if t not in cc:
            n, lab, st, _ = cv2.connectedComponentsWithStats((prob > t).astype(np.uint8), connectivity=8)
            peak = _peaks(prob, lab, n)
            cc[t] = (lab, st[:, cv2.CC_STAT_AREA], peak)
        lab, area, peak = cc[t]
        keep = (area >= b["min_area"]) & (peak > b["seed_thresh"])
        keep[0] = False
        key = (t, b["group_radius"], b["min_area"], keep.tobytes())
        if key not in memo:
            inst = relabel(group_disk(keep[lab].astype(np.uint8), b["group_radius"]), b["min_area"])
            memo[key] = stats(inst, gts)
        rows.append(memo[key])
    return np.stack(rows)


def _members(stem):
    """(N1 upsampled, N2) float maps of `stem` from MEMBERS (lists of dirs, one per fold)."""
    out = []
    for dirs in MEMBERS:
        path = next(Path(d) / f"{stem}.npy" for d in dirs if (Path(d) / f"{stem}.npy").exists())
        m = np.load(path).astype(np.float32)
        if m.shape[0] != 2048:
            m = cv2.resize(m, (2048, 2048), interpolation=cv2.INTER_LINEAR)
        out.append(m / 255.0)
    return out


MEMBERS: list[list[str]] = []


def _job_var(args):
    probs, stem, rles, tasks, members = args   # tasks: [(base dict, variant, extra dict)]
    global MEMBERS
    MEMBERS = members
    prob, gts = _load(probs, stem), _gts(rles)
    mem = _members(stem) if any(v == "extmix" for _, v, _ in tasks) else None
    return np.stack([stats(build(prob, b, v, e, mem), gts) for b, v, e in tasks])


def pq(s):
    den = s[1] + 0.5 * s[2] + 0.5 * s[3]
    return s[0] / den if den else 0.0


def splits(n, repeats=3):
    for r in range(repeats):
        half = np.random.default_rng(100 + r).permutation(n) < n // 2
        yield half
        yield ~half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probs")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help="first N images only (smoke test)")
    ap.add_argument("--members", help="extmix: comma-separated N1 dirs, ':' then N2 dirs")
    ap.add_argument("--save", type=Path, help="npz with per-combo per-image stats")
    args = ap.parse_args()
    t0 = time.time()

    by: dict[str, list] = {}
    for it in with_maps(load_train(), Path(args.probs)):
        by.setdefault(it.image_id, []).append(it.gt_rles)
    stems = sorted(by)[: args.limit or None]
    res = np.load(Path(args.probs) / f"{stems[0]}.npy").shape[0]
    s = res / 2048
    base = [_scale(c, s) for c in combos(BASE)]
    with ProcessPoolExecutor(args.jobs) as pool:
        per_base = np.stack(list(pool.map(_job_base, [(args.probs, st, by[st], base) for st in stems],
                                          chunksize=2)), 1)   # (n_base, n_img, 4)
    print(f"base grid done {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    sp = list(splits(len(stems)))
    chosen = [int(np.argmax([pq(per_base[c][a].sum(0)) for c in range(len(base))])) for a in sp]
    base_out = np.zeros((len(stems), 4))
    for a, c in zip(sp, chosen):
        base_out[~a] += per_base[c][~a]
    base_out /= len(sp)
    print(f"baseline chosen: {[base[c] for c in sorted(set(chosen))]}", file=sys.stderr, flush=True)

    variants = args.variants.split(",")
    members = [g.split(",") for g in args.members.split(":")] if args.members else []
    extras = {v: [_scale(e, s) for e in VARIANTS[v]] for v in variants}
    tasks = [(base[c], v, e) for c in sorted(set(chosen)) for v in variants for e in extras[v]]
    index = {(c, v, j): i for i, (c, v, j) in enumerate(
        (c, v, j) for c in sorted(set(chosen)) for v in variants for j in range(len(extras[v])))}
    with ProcessPoolExecutor(args.jobs) as pool:
        per_task = np.stack(list(pool.map(_job_var, [(args.probs, st, by[st], tasks, members) for st in stems],
                                          chunksize=2)), 1)   # (n_tasks, n_img, 4)
    print(f"variants done {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    outs, picks = {}, {}
    for v in variants:
        o = np.zeros((len(stems), 4))
        pk = []
        for a, c in zip(sp, chosen):
            j = int(np.argmax([pq(per_task[index[(c, v, j)]][a].sum(0)) for j in range(len(extras[v]))]))
            pk.append(extras[v][j])
            o[~a] += per_task[index[(c, v, j)]][~a]
        outs[v], picks[v] = o / len(sp), pk
    if args.save:
        np.savez_compressed(args.save, stems=np.array(stems), per_base=per_base, per_task=per_task,
                            base_out=base_out, **{f"out_{v}": o for v, o in outs.items()})

    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(stems), size=(args.boot, len(stems)))
    bb = np.array([pq(base_out[b].sum(0)) for b in boot])
    print(f"{args.probs}: {len(stems)} images, {args.boot} bootstrap resamples, res {res}")
    print(f"{'pipeline':12s} {'PQ':>7s} {'TP':>7s} {'FP':>7s} {'FN':>7s} {'diff':>8s} {'95% CI':>18s} {'P>0':>5s}")
    tot = base_out.sum(0)
    print(f"{'hyst_soft':12s} {pq(tot):7.4f} {tot[1]:7.1f} {tot[2]:7.1f} {tot[3]:7.1f}")
    for v in variants:
        tot = outs[v].sum(0)
        d = np.array([pq(outs[v][b].sum(0)) for b in boot]) - bb
        print(f"{v:12s} {pq(tot):7.4f} {tot[1]:7.1f} {tot[2]:7.1f} {tot[3]:7.1f} {pq(tot) - pq(base_out.sum(0)):+8.4f}"
              f" [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] {np.mean(d > 0):5.2f}")
    for v in variants:
        print(f"  {v} picks per split: {picks[v]}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
