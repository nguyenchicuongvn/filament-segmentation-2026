"""Training-lens error anatomy on cached val maps (read-only w.r.t. repo).

For one probs dir + post config:
 A. per prediction instance: matched by >=1 annotator ('hit') or none ('fp0'); peak/mean prob, area,
    r/R (disk radius), distance to nearest GT pixel of any annotator (fragment vs isolated).
 B. per GT: best IoU, max/mean prob inside GT (map res), size, support (#other annotators matching).
 C. human-vs-human recall by GT size bin (all multi-annotator train images) -> irreducible baseline.
Usage: python fp_fn_anatomy.py <probs_dir> <post.json> <tag>
"""
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from filseg.classical import fit_disk  # noqa: E402
from filseg.data import by_image, load_gray, load_train, with_maps  # noqa: E402
from filseg.dl import PostParams, instances_lowres, load_maps  # noqa: E402

PROBS = Path(sys.argv[1])
POST = PostParams(**json.loads(Path(sys.argv[2]).read_text()))
TAG = sys.argv[3]
OUT = Path(__file__).parent / f"anatomy_{TAG}.npz"
BINS = [(0, 400), (400, 800), (800, 1600), (1600, 3200), (3200, 8000), (8000, 1e9)]


def iou(a, b):
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return np.asarray(mu.iou(a, b, [0] * len(b)))


def job(args):
    stem, path, entries = args
    prob, group = load_maps(PROBS, stem)
    res = prob.shape[0]
    s = 2048 / res
    inst = instances_lowres(prob, POST, group)
    n = int(inst.max())
    up = cv2.resize(inst.astype(np.float32), (2048, 2048), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    pred = [mu.encode(np.asfortranarray((up == k).astype(np.uint8))) for k in range(1, n + 1)]
    cx, cy, R = fit_disk(load_gray(path))
    cx, cy, R = cx / s, cy / s, R / s
    yy, xx = np.mgrid[:res, :res]
    rr = np.hypot(yy - cy, xx - cx) / R
    # union of all annotators' GT at map res, distance transform to it
    anyg = np.zeros((res, res), np.uint8)
    gts_low = {}
    for key, gts in entries:
        lows = []
        for r in gts:
            m = cv2.resize(mu.decode(r).astype(np.float32), (res, res), interpolation=cv2.INTER_AREA) >= 0.25
            lows.append(m)
            anyg |= m
        gts_low[key] = lows
    dist = cv2.distanceTransform((1 - anyg).astype(np.uint8), cv2.DIST_L2, 3)
    hits = np.zeros(n, int)
    preds_rows = []
    gt_rows = []
    for key, gts in entries:
        m = iou(gts, pred)  # (ngt, npred)
        if m.size:
            hits += (m > 0.5).any(0)
        others = [g for k, g in entries if k != key]
        sup = np.zeros(len(gts), int)
        for og in others:
            if gts and og:
                sup += (iou(gts, og).max(1) > 0.5)
        garea = mu.area(gts) if gts else np.zeros(0)
        best = m.max(1) if m.size else np.zeros(len(gts))
        for i, gl in enumerate(gts_low[key]):
            pv = prob[gl] if gl.any() else np.array([0.0])
            gt_rows.append((len(entries), garea[i], best[i], pv.max(), pv.mean(), sup[i],
                            rr[gl].mean() if gl.any() else np.nan, key[:2] == "01"))
    for k in range(1, n + 1):
        msk = inst == k
        pv = prob[msk]
        preds_rows.append((len(entries), hits[k - 1], msk.sum() * s * s, pv.max(), pv.mean(), rr[msk].mean(),
                           dist[msk].min(), np.percentile(dist[msk], 50)))
    return preds_rows, gt_rows


def human(args):
    stem, entries = args
    out = []
    for k1, g1 in entries:
        a1 = mu.area(g1) if g1 else np.zeros(0)
        for k2, g2 in entries:
            if k1 != k2:
                m = iou(g1, g2)
                best = m.max(1) if m.size else np.zeros(len(g1))
                for i in range(len(g1)):
                    out.append((len(entries), a1[i], best[i]))
    return out


def main():
    items = load_train()
    val = with_maps(items, PROBS)
    groups = [(s, es[0].path, [(t.key, t.gt_rles) for t in es]) for s, es in by_image(val).items()]
    allg = [(s, [(t.key, t.gt_rles) for t in es]) for s, es in by_image(items).items() if len(es) > 1]
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(job, groups, chunksize=4))
        hum = [r for rs in ex.map(human, allg, chunksize=8) for r in rs] if TAG == "effb4" else []
    P = np.array([p for r in res for p in r[0]], float)
    G = np.array([g for r in res for g in r[1]], float)
    np.savez(OUT, P=P, G=G, H=np.array(hum, float))
    print("saved", OUT, P.shape, G.shape)


if __name__ == "__main__":
    main()
