"""Per-instance input-filter features on E9a holdout (213 imgs, hyst config).

For each prediction: mean matched IoU over the image's annotators (F26 label), any-TP, and
features from the model map (pmean, pmax, area, elong, r) and from the image (flat-fielded
intensity, contrast vs ring, black-hat, Sato ridge (dark), DoG). For each GT: detected?, support
by other annotators, same image features. Per entry: PQ stats, station letter. Writes feats.pkl.
"""
import json, os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from filseg.classical import fit_disk, radial_profile  # noqa
from filseg.data import by_image, encode_masks, load_gray, load_train, with_maps  # noqa
from filseg.dl import PostParams, instances_lowres, load_maps, upsample_labels  # noqa

PROBS = ROOT / "outputs/colab/unet1024_effb4/probs_val"
POST = PostParams(**json.loads((ROOT / "configs/post_unet1024_effb4_hyst.json").read_text()))
OUT = Path(__file__).parent / "feats.pkl"
R = 1024


def iou(gt, pr):
    if not gt or not pr:
        return np.zeros((len(gt), len(pr)))
    return np.asarray(mu.iou(pr, gt, [0] * len(gt))).T


def image_maps(gray):
    g = cv2.resize(gray, (R, R), interpolation=cv2.INTER_AREA).astype(np.float32)
    cx, cy, dr = fit_disk(g.astype(np.uint8))
    yy, xx = np.mgrid[:R, :R]
    rr = np.hypot(xx - cx, yy - cy)
    ind = rr < dr
    prof = np.maximum(radial_profile(g, rr, dr, ind), 1.0)
    flat = np.where(ind, g / prof, 1.0).astype(np.float32)
    # large-scale flat field (residual)
    bg = cv2.medianBlur(np.clip(flat * 127, 0, 255).astype(np.uint8), 61).astype(np.float32) / 127
    flat2 = np.where(ind, flat / np.maximum(bg, 0.2), 1.0).astype(np.float32)
    bh = cv2.morphologyEx(flat2, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    dog = cv2.GaussianBlur(flat2, (0, 0), 8) - cv2.GaussianBlur(flat2, (0, 0), 1.5)
    from skimage.filters import sato
    sat = sato(flat2, sigmas=(1, 2, 3), black_ridges=True).astype(np.float32)
    sat[~ind] = 0
    # image-level stats
    inner = rr < 0.9 * dr
    lap = cv2.Laplacian(flat2, cv2.CV_32F)
    hp = flat2 - cv2.GaussianBlur(flat2, (0, 0), 1.0)
    st = dict(disk_r=dr, raw_med=float(np.median(g[inner])), raw_p5=float(np.percentile(g[inner], 5)),
              raw_p95=float(np.percentile(g[inner], 95)),
              limb_ratio=float(prof[(rr > 0.95 * dr) & ind].mean() / prof[rr < 0.2 * dr].mean()),
              flat_std=float(flat2[inner].std()), lap_var=float(lap[inner].var()),
              hp_std=float(hp[inner].std()), sat_mean=float(sat[inner].mean()),
              bh_p99=float(np.percentile(bh[inner], 99)))
    return dict(disk=(cx, cy, dr), flat=flat2, bh=bh, dog=dog, sat=sat), st


def feats(mask, ctx, prob):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    cx, cy, dr = ctx["disk"]
    r = float(np.hypot(xs.mean() - cx, ys.mean() - cy) / dr)
    if len(ys) > 2:
        ev = np.sort(np.linalg.eigvalsh(np.cov(np.stack([xs, ys]).astype(np.float64))))
        elong = float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
    else:
        elong = 1.0
    y0, y1, x0, x1 = max(0, ys.min() - 12), ys.max() + 13, max(0, xs.min() - 12), xs.max() + 13
    sub = mask[y0:y1, x0:x1].astype(np.uint8)
    ring = cv2.dilate(sub, np.ones((11, 11), np.uint8)).astype(bool) & ~sub.astype(bool)
    fl = ctx["flat"][y0:y1, x0:x1]
    fin = float(fl[sub > 0].mean())
    fout = float(np.median(fl[ring])) if ring.any() else fin
    satsub = ctx["sat"][y0:y1, x0:x1]
    return dict(area=len(ys), r=r, elong=elong, pmean=float(prob[ys, xs].mean()), pmax=float(prob[ys, xs].max()),
                p90=float(np.percentile(prob[ys, xs], 90)),
                flat=fin, flatmin=float(np.percentile(fl[sub > 0], 5)), contrast=fin / max(fout, 1e-3),
                bh=float(ctx["bh"][ys, xs].mean()), bhmax=float(np.percentile(ctx["bh"][ys, xs], 90)),
                dog=float(ctx["dog"][ys, xs].mean()), sat=float(ctx["sat"][ys, xs].mean()),
                satmax=float(np.percentile(ctx["sat"][ys, xs], 90)),
                sat_ratio=float(satsub[sub > 0].mean() / max(satsub[ring].mean(), 1e-6)) if ring.any() else 1.0)


def lr(rle):
    m = mu.decode(rle)
    return cv2.resize(m, (R, R), interpolation=cv2.INTER_AREA) >= 0.5


def job(args):
    stem, entries = args
    prob, group = load_maps(PROBS, stem)
    inst = instances_lowres(prob, POST, group)
    pred = encode_masks(upsample_labels(inst))
    ctx, ist = image_maps(load_gray(entries[0][2]))
    np_ = len(pred)
    pf = [feats(inst == k, ctx, prob) for k in range(1, np_ + 1)]
    ious = [iou(g, pred) for _, g, _ in entries]
    mm = np.zeros(np_)
    anytp = np.zeros(np_, bool)
    for m in ious:
        if m.size:
            best = m.max(0)
            mm += np.where(best > 0.5, best, 0)
            anytp |= best > 0.5
    mm /= len(entries)
    preds = [dict(stem=stem, nann=len(entries), miou=mm[j], anytp=bool(anytp[j]), **pf[j])
             for j in range(np_) if pf[j] is not None]
    gts, ent = [], []
    for (key, g, _), m in zip(entries, ious):
        hit = m > 0.5
        ent.append(dict(key=key, stem=stem, station=stem[-2], nann=len(entries),
                        siou=float(m[hit].sum()), tp=int(hit.sum()),
                        fp=int((hit.sum(0) == 0).sum()) if m.size else np_,
                        fn=int((hit.sum(1) == 0).sum()) if m.size else len(g)))
        for i, rle in enumerate(g):
            sup = 0
            for (k2, g2, _) in entries:
                if k2 != key and g2:
                    sup += iou([rle], g2).max() > 0.5
            f = feats(lr(rle), ctx, prob)
            if f is None:
                continue
            gts.append(dict(stem=stem, key=key, nann=len(entries), sup=int(sup), det=bool(m[i].max() > 0.5) if m.size else False,
                            best=float(m[i].max()) if m.size else 0.0, area2048=float(mu.area([rle])[0]), **f))
    return dict(stem=stem, station=stem[-2], img=ist, preds=preds, gts=gts, ent=ent)


if __name__ == "__main__":
    items = with_maps(load_train(), PROBS)
    groups = by_image(items)
    jobs = [(s, [(t.key, t.gt_rles, t.path) for t in es]) for s, es in groups.items()]
    if len(sys.argv) > 1:
        jobs = jobs[: int(sys.argv[1])]
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(job, jobs, chunksize=2))
    pickle.dump(res, open(OUT, "wb"))
    print("images", len(res), "preds", sum(len(r["preds"]) for r in res), "gts", sum(len(r["gts"]) for r in res))
