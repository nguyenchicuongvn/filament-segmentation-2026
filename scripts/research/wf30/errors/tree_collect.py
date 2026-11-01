"""Component-tree candidates on E9a val maps.

Levels t in LEVELS; at each level the candidates are the grouped components (dilate r=GR, as the
base post) of prob > t. Components are nested across levels -> a tree. For each candidate: features
and, per annotator entry, the best IoU with that entry's GTs (computed at 2048 like the host).
With a fixed instance set PQ's denominator is 0.5 * sum_e (n_pred + n_gt_e), so a candidate's
contribution is separable: value = sum_e matched IoU (IoU>0.5), cost = 0.5 * n_entries per instance.
"""
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
from filseg.dl import load_maps, upsample_labels, _disk  # noqa: E402

PROBS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/colab/unet1024_effb4/probs_val"
GR = int(sys.argv[2]) if len(sys.argv) > 2 else 4
OUT = Path(__file__).parent / f"tree_{PROBS.parent.name}_r{GR}.pkl"
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
MIN_AREA = 20


def job(args):
    stem, entries = args
    prob, _ = load_maps(PROBS, stem)
    R = prob.shape[0]
    gray = load_gray(entries[0][2])
    g = cv2.resize(gray, (R, R), interpolation=cv2.INTER_AREA).astype(np.float32)
    cx, cy, dr = fit_disk(g.astype(np.uint8))
    yy, xx = np.mgrid[:R, :R]
    rrm = np.hypot(xx - cx, yy - cy)
    ind = rrm < dr
    flat = np.where(ind, g / np.maximum(radial_profile(g, rrm, dr, ind), 1.0), 1.0)
    cands = []
    prev_lab = None
    prev_ids = None
    for li, t in enumerate(LEVELS):
        fg = (prob > t).astype(np.uint8)
        grouped = cv2.dilate(fg, _disk(GR)) if GR > 0 else fg
        _, glab = cv2.connectedComponents(grouped, connectivity=8)
        lab = np.where(fg > 0, glab, 0).astype(np.int32)
        areas = np.bincount(lab.ravel())
        keep = areas >= MIN_AREA
        keep[0] = False
        lut = np.zeros(len(areas), np.int32)
        lut[keep] = np.arange(1, keep.sum() + 1)
        lab = lut[lab]
        n = int(lab.max())
        if n == 0:
            break
        rles = encode_masks(upsample_labels(lab))
        ids = []
        objs = __import__("scipy.ndimage", fromlist=["find_objects"]).find_objects(lab)
        for k in range(1, n + 1):
            sl = objs[k - 1]
            m = lab[sl] == k
            ys, xs = np.nonzero(m)
            ys = ys + sl[0].start; xs = xs + sl[1].start
            p = prob[ys, xs]
            parent = -1
            if prev_lab is not None:
                pl = prev_lab[ys[0], xs[0]]
                parent = prev_ids[pl - 1] if pl > 0 else -1
            if len(ys) > 2:
                ev = np.sort(np.linalg.eigvalsh(np.cov(np.stack([xs, ys]).astype(np.float64))))
                elong = float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
                major = float(4 * np.sqrt(max(ev[1], 0)))
            else:
                elong, major = 1.0, 1.0
            y0, y1, x0, x1 = max(0, ys.min() - 10), ys.max() + 11, max(0, xs.min() - 10), xs.max() + 11
            sub = (lab[y0:y1, x0:x1] == k).astype(np.uint8)
            ring = cv2.dilate(sub, np.ones((9, 9), np.uint8)).astype(bool) & (prob[y0:y1, x0:x1] < 0.1)
            fl = flat[y0:y1, x0:x1]
            fin = float(fl[sub > 0].mean())
            fout = float(np.median(fl[ring])) if ring.any() else fin
            nfrag = cv2.connectedComponents(sub, connectivity=8)[0] - 1
            mx, my = xs.mean(), ys.mean()
            x, y = (mx - cx) / dr, (my - cy) / dr
            feats = dict(level=t, log_area=np.log(len(ys)), pmean=float(p.mean()), pmax=float(p.max()),
                         p25=float(np.percentile(p, 25)), p75=float(np.percentile(p, 75)),
                         frac90=float((p > 0.9).mean()), elong=elong, major=major,
                         extent=len(ys) / float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)),
                         contrast=fin / max(fout, 1e-3), flat=fin, nfrag=nfrag,
                         r=float(np.hypot(x, y)), absx=abs(float(x)), absy=abs(float(y)))
            gi = []
            for key, gts in entries_gts(entries):
                if gts:
                    iou = np.asarray(mu.iou([rles[k - 1]], gts, [0] * len(gts)))[0]
                    gi.append(float(iou.max()))
                else:
                    gi.append(0.0)
            cands.append(dict(stem=stem, li=li, parent=parent, area=len(ys), ious=gi, **feats))
            ids.append(len(cands) - 1)
        prev_lab, prev_ids = lab, ids
    ngt = [len(gts) for _, gts in entries_gts(entries)]
    # local index -> keep parent as local index within this image
    return dict(stem=stem, cands=cands, ngt=ngt, keys=[e[0] for e in entries])


def entries_gts(entries):
    return [(k, g) for k, g, _ in entries]


if __name__ == "__main__":
    items = with_maps(load_train(), PROBS)
    groups = by_image(items)
    jobs = [(s, [(t.key, t.gt_rles, t.path) for t in es]) for s, es in groups.items()]
    if len(sys.argv) > 3:
        jobs = jobs[: int(sys.argv[3])]
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(job, jobs, chunksize=2))
    pickle.dump(res, open(OUT, "wb"))
    print("images", len(res), "cands", sum(len(r["cands"]) for r in res))
