"""Per-object error forensics on E9a holdout maps. Writes records.pkl.

For every annotator entry: each prediction and each GT gets a row with its match status, overlap
structure (split/merge/near-miss) and descriptors (disk position, size, shape, contrast, darkness,
sunspot proximity, prob stats). Work at 2048 for IoU (as the host), descriptors at 1024.
"""
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from filseg.classical import fit_disk, radial_profile  # noqa: E402
from filseg.data import by_image, encode_masks, load_gray, load_train, with_maps  # noqa: E402
from filseg.dl import PostParams, instances_lowres, load_maps, upsample_labels  # noqa: E402

PROBS = ROOT / "outputs/colab/unet1024_effb4/probs_val"
POST = PostParams(**json.loads((ROOT / "configs/post_unet1024_effb4_hyst.json").read_text()))
OUT = Path(__file__).parent / "records.pkl"
R = 1024


def iou(a, b):
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return np.asarray(mu.iou(b, a, [0] * len(a))).T  # (len(a), len(b))  a=gt, b=pred


def inter(a, b):
    """(len(a), len(b)) intersection areas."""
    if not a or not b:
        return np.zeros((len(a), len(b)))
    m = iou(a, b)
    A, B = mu.area(a)[:, None].astype(float), mu.area(b)[None, :].astype(float)
    return m * (A + B) / (1 + m)


def descr(mask_lr, ctx):
    """descriptors of a 1024 mask."""
    ys, xs = np.nonzero(mask_lr)
    if len(ys) == 0:
        return dict(r=np.nan, lon=np.nan, lat=np.nan, area_lr=0, elong=np.nan, major=np.nan,
                    contrast=np.nan, flat=np.nan, flatmin=np.nan, sun_d=np.nan, pmean=np.nan, pmax=np.nan)
    cx, cy, dr = ctx["disk"]
    mx, my = xs.mean(), ys.mean()
    x, y = (mx - cx) / dr, (my - cy) / dr
    rr = float(np.hypot(x, y))
    lon = float(np.degrees(np.arcsin(np.clip(x / max(np.sqrt(max(1 - y * y, 1e-6)), 1e-6), -1, 1))))
    lat = float(np.degrees(np.arcsin(np.clip(-y, -1, 1))))
    if len(ys) > 2:
        ev = np.sort(np.linalg.eigvalsh(np.cov(np.stack([xs, ys]).astype(np.float64))))
        elong = float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
        major = float(4 * np.sqrt(max(ev[1], 0)))
    else:
        elong, major = 1.0, 1.0
    y0, y1, x0, x1 = max(0, ys.min() - 12), ys.max() + 13, max(0, xs.min() - 12), xs.max() + 13
    sub = mask_lr[y0:y1, x0:x1].astype(np.uint8)
    ring = cv2.dilate(sub, np.ones((9, 9), np.uint8)).astype(bool) & ~sub.astype(bool)
    fl = ctx["flat"][y0:y1, x0:x1]
    fin = float(fl[sub > 0].mean())
    fout = float(np.median(fl[ring])) if ring.any() else fin
    sd = ctx["sun_dt"]
    return dict(r=rr, lon=lon, lat=lat, area_lr=int(len(ys)), elong=elong, major=major,
                contrast=fin / max(fout, 1e-3), flat=fin, flatmin=float(np.percentile(fl[sub > 0], 5)),
                sun_d=float(sd[ys, xs].min()) / dr if sd is not None else np.nan,
                pmean=float(ctx["prob"][ys, xs].mean()), pmax=float(ctx["prob"][ys, xs].max()))


def lr(rle):
    m = mu.decode(rle)
    return cv2.resize(m, (R, R), interpolation=cv2.INTER_AREA) >= 0.5 if m.shape[0] != R else m.astype(bool)


def job(args):
    stem, entries = args
    prob, group = load_maps(PROBS, stem)
    inst = instances_lowres(prob, POST, group)
    pred = encode_masks(upsample_labels(inst))
    np_ = len(pred)
    parea = mu.area(pred) if pred else np.zeros(0)
    gray = load_gray(entries[0][2])
    g = cv2.resize(gray, (R, R), interpolation=cv2.INTER_AREA).astype(np.float32)
    cx, cy, dr = fit_disk(g.astype(np.uint8))
    yy, xx = np.mgrid[:R, :R]
    rrm = np.hypot(xx - cx, yy - cy)
    ind = rrm < dr
    flat = g / np.maximum(radial_profile(g, rrm, dr, ind), 1.0)
    flat = np.where(ind, flat, 1.0)
    # sunspot proxy: very dark compact cores (flat < 0.55), area 3..2000 px at 1024, inside 0.97 R
    dark = ((flat < 0.55) & (rrm < 0.97 * dr)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    spots = np.zeros_like(dark)
    for k in range(1, n):
        a = st[k, cv2.CC_STAT_AREA]
        w, h = st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]
        if 3 <= a <= 2000 and max(w, h) / max(1, min(w, h)) < 3 and a / (w * h) > 0.35:
            spots[lab == k] = 1
    sun_dt = cv2.distanceTransform((1 - spots).astype(np.uint8), cv2.DIST_L2, 5) if spots.any() else None
    ctx = dict(disk=(cx, cy, dr), flat=flat, sun_dt=sun_dt, prob=prob)
    pdesc = [descr(inst == k, ctx) for k in range(1, np_ + 1)]
    gt_lr_cache = {}
    all_gts = {key: gts for key, gts, _ in entries}
    out = []
    for key, gts, _ in entries:
        ng = len(gts)
        garea = mu.area(gts) if gts else np.zeros(0)
        m = iou(gts, pred)
        I = inter(gts, pred)
        # other annotators' match of each pred
        other_match = np.zeros(np_, bool)
        for k2, g2 in all_gts.items():
            if k2 != key and g2 and pred:
                other_match |= iou(g2, pred).max(0) > 0.5
        sup = np.zeros(ng, int)
        for k2, g2 in all_gts.items():
            if k2 != key and g2 and gts:
                sup += iou(gts, g2).max(1) > 0.5
        preds = []
        for j in range(np_):
            col = m[:, j] if ng else np.zeros(0)
            best = float(col.max()) if ng else 0.0
            bi = int(col.argmax()) if ng else -1
            covfrac = I[:, j] / max(parea[j], 1) if ng else np.zeros(0)
            preds.append(dict(key=key, stem=stem, j=j, area=float(parea[j]), best=best, bi=bi,
                              tp=best > 0.5, other=bool(other_match[j]),
                              ncov=int((covfrac > 0.1).sum()) if ng else 0,
                              cov_best=float(covfrac.max()) if ng else 0.0,
                              in_gt=float(covfrac.sum()) if ng else 0.0,
                              ratio=float(parea[j] / garea[bi]) if ng else np.nan, **pdesc[j]))
        gtsr = []
        for i in range(ng):
            row = m[i]
            best = float(row.max()) if np_ else 0.0
            bj = int(row.argmax()) if np_ else -1
            cov = I[i] / max(garea[i], 1) if np_ else np.zeros(0)
            touching = [j for j in range(np_) if cov[j] > 0.05 or (I[i, j] / max(parea[j], 1)) > 0.3]
            uni = np.nan
            if len(touching) >= 2:
                u = mu.merge([pred[j] for j in touching])
                uni = float(mu.iou([u], [gts[i]], [0])[0][0])
            # merge: best pred also covers other GTs substantially
            merged = False
            if np_ and best > 0.1:
                cf = I[:, bj] / max(parea[bj], 1)
                merged = int((cf > 0.1).sum()) >= 2
            ck = (key, i)
            gl = lr(gts[i])
            d = descr(gl, ctx)
            gtsr.append(dict(key=key, stem=stem, i=i, area=float(garea[i]), best=best, bj=bj,
                             tp=best > 0.5, sup=int(sup[i]), nann=len(entries), ntouch=len(touching),
                             union_iou=uni, touching=touching, merged=merged,
                             cov=float(cov.max()) if np_ else 0.0,
                             ratio=float(parea[bj] / garea[i]) if np_ else np.nan, **d))
        out.append(dict(key=key, stem=stem, nann=len(entries), preds=preds, gts=gtsr,
                        tp_iou=m[m > 0.5].tolist() if m.size else []))
    return out


if __name__ == "__main__":
    items = with_maps(load_train(), PROBS)
    groups = by_image(items)
    jobs = [(s, [(t.key, t.gt_rles, t.path) for t in es]) for s, es in groups.items()]
    if len(sys.argv) > 1:
        jobs = jobs[: int(sys.argv[1])]
    res = []
    with ProcessPoolExecutor(4) as ex:
        for r in ex.map(job, jobs, chunksize=2):
            res.extend(r)
    pickle.dump(res, open(OUT, "wb"))
    print("entries", len(res))
