"""Option (b) prototype: learned pairwise grouping of hysteresis components (CPU), N1+N2 OOF maps.

Candidate pairs = components whose gap <= MAXGAP px (2048 scale). Label = consensus GT relation
(same filament / different filament, as in spine_group_oracle.py); unassigned pairs are not trained on.
Features: gap, both areas, min / mean probability along the straight segment joining the closest
points, principal-axis angle between components, angle of the gap to each axis, elongation of each,
distance from the closest points to each component's far end (endpoint-ness).
GBM, 5-fold by image (fold_split folds), out-of-fold join probabilities; the join threshold is
cross-tuned (half A chooses, half B scored, both ways, 3 splits). Pairs with p > t are joined (union-find);
pairs below t are NOT joined even if the radius rule would join them.
v2 (--v2, fixed 05/11 before scoring): gap widened to 64 px, plus local-tangent features at the contact
points (PCA of each component's pixels within 24 px of its contact point): |cos| of each local tangent
with the gap vector, |cos| between the two local tangents, and how far each contact point misses the other
component's tangent line (px). Acceptance (pre-registered): v2 is taken only if its cross-tuned delta beats
v1's; it ships only if the paired bootstrap 95% CI is above 0 and delta > 0 on >= 4 of 5 folds.
Usage: .venv/bin/python scripts/research/spine_pair_group.py [--jobs 8] [--v2]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from filseg.data import by_image, encode_masks, fold_split, load_train  # noqa: E402
from filseg.metric import PQAccumulator  # noqa: E402
from spine_group_oracle import CFG, MAPS, UF, disk  # noqa: E402

MAXGAP = 48
LOCAL_R = 24


def comp_geom(mask):
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    c = pts.mean(0)
    cov = np.cov((pts - c).T) if len(pts) > 2 else np.eye(2)
    ev, evec = np.linalg.eigh(cov)
    axis = evec[:, 1]
    elong = float(np.sqrt(max(ev[1], 1e-6) / max(ev[0], 1e-6)))
    proj = (pts - c) @ axis
    return pts, c, axis, elong, proj.min(), proj.max()


def feats_image(args):
    stem, gts, v2 = args
    maxgap = 64 if v2 else MAXGAP
    prob = np.load(MAPS / f"{stem}.npy").astype(np.float32) / 255.0
    fg = (prob > CFG["thresh"]).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    peak = np.zeros(n, np.float32)
    np.maximum.at(peak, lab.ravel(), prob.ravel())
    keep = (st[:, cv2.CC_STAT_AREA] >= CFG["min_area"]) & (peak > CFG["seed_thresh"])
    keep[0] = False
    ids = np.nonzero(keep)[0]
    lut = np.zeros(n, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    comp = lut[lab]
    m = len(ids)
    area = np.bincount(comp.ravel(), minlength=m + 1)
    # radius rule
    grown = cv2.dilate((comp > 0).astype(np.uint8), disk(CFG["group_radius"]))
    _, glab = cv2.connectedComponents(grown, connectivity=8)
    grp = np.zeros(m + 1, np.int64)
    pix = comp > 0
    grp[comp[pix]] = glab[pix]
    # GT relation
    assign = np.full((len(gts), m + 1), -1, np.int64)
    for e, rles in enumerate(gts):
        if rles:
            g = mu.decode(rles)
            for k in range(g.shape[2]):
                ov = np.bincount(comp[g[:, :, k] > 0], minlength=m + 1)
                hit = ov >= 0.5 * area
                hit[0] = False
                assign[e, hit] = k
    geo = {}
    boxes = {}
    for i in range(1, m + 1):
        x, y, w, h = st[ids[i - 1], :4]
        boxes[i] = (x, y, w, h)
    rows, pairs = [], []
    for i in range(1, m + 1):
        xi, yi, wi, hi = boxes[i]
        for j in range(i + 1, m + 1):
            xj, yj, wj, hj = boxes[j]
            if xj > xi + wi + maxgap or xi > xj + wj + maxgap or yj > yi + hi + maxgap or yi > yj + hj + maxgap:
                continue
            for k in (i, j):
                if k not in geo:
                    x, y, w, h = boxes[k]
                    sub = comp[y:y + h, x:x + w] == k
                    pts, c, ax, el, pmin, pmax = comp_geom(sub)
                    geo[k] = (pts + [x, y], c + [x, y], ax, el, pmin, pmax)
            pi, ci, ai, ei, mi0, mi1 = geo[i]
            pj, cj, aj, ej, mj0, mj1 = geo[j]
            # closest points (subsample for speed)
            si = pi[:: max(1, len(pi) // 400)]
            sj = pj[:: max(1, len(pj) // 400)]
            d = np.sqrt(((si[:, None] - sj[None]) ** 2).sum(-1))
            a, b = np.unravel_index(np.argmin(d), d.shape)
            gap = float(d[a, b])
            if gap > maxgap:
                continue
            p0, p1 = si[a], sj[b]
            L = max(2, int(np.ceil(gap)) + 1)
            line = np.linspace(p0, p1, L)
            v = prob[np.clip(line[:, 1].round().astype(int), 0, prob.shape[0] - 1),
                     np.clip(line[:, 0].round().astype(int), 0, prob.shape[1] - 1)]
            gv = (p1 - p0) / max(gap, 1e-3)
            ang = abs(float(ai @ aj))
            gi = abs(float(gv @ ai)) if gap > 0 else 1.0
            gj = abs(float(gv @ aj)) if gap > 0 else 1.0
            # endpoint-ness: how close the contact point is to an end of the principal axis (0 = end, 1 = middle)
            ti = (float((p0 - ci) @ ai) - mi0) / max(mi1 - mi0, 1)
            tj = (float((p1 - cj) @ aj) - mj0) / max(mj1 - mj0, 1)
            endi, endj = 1 - 2 * abs(ti - 0.5), 1 - 2 * abs(tj - 0.5)
            both = (assign[:, i] >= 0) & (assign[:, j] >= 0)
            nb = int(both.sum())
            same = int((assign[both, i] == assign[both, j]).sum())
            y = -1 if nb == 0 else int(2 * same >= nb)
            rows.append([gap, np.log(area[i]), np.log(area[j]), np.log(min(area[i], area[j])),
                         float(v.min()), float(v.mean()), ang, gi, gj, min(gi, gj), ei, ej, min(endi, endj),
                         max(endi, endj), float(grp[i] == grp[j]),
                         float(prob[comp == i].mean()), float(prob[comp == j].mean())]
                        + (local_feats(pi, p0, pj, p1, gv) if v2 else []))
            pairs.append((i, j, y))
    return stem, comp, np.asarray(rows, np.float32).reshape(-1, 22 if v2 else 17), pairs, grp


def local_axis(pts, p):
    near = pts[((pts - p) ** 2).sum(1) <= LOCAL_R ** 2]
    if len(near) < 3:
        return None
    ev, evec = np.linalg.eigh(np.cov((near - near.mean(0)).T))
    return evec[:, 1]


def local_feats(pi, p0, pj, p1, gv):
    """Local-tangent continuity of the gap (v2)."""
    ti, tj = local_axis(pi, p0), local_axis(pj, p1)
    if ti is None or tj is None:
        return [-1.0, -1.0, -1.0, -1.0, -1.0]
    d = p1 - p0
    miss_i = abs(float(d[0] * ti[1] - d[1] * ti[0]))   # distance of p1 from p0's tangent line
    miss_j = abs(float(d[0] * tj[1] - d[1] * tj[0]))
    return [abs(float(gv @ ti)), abs(float(gv @ tj)), abs(float(ti @ tj)), miss_i, miss_j]


_CACHE = {}


def _img_stats(s, data, gts_by, joins):
    key = (s, tuple(joins))
    if key in _CACHE:
        return _CACHE[key]
    comp, pairs, grp = data[s]
    m = int(comp.max())
    uf = UF(m + 1)
    for (i, j, _), jn in zip(pairs, joins):
        if jn:
            uf.u(i, j)
    root = np.array([uf.f(i) for i in range(m + 1)])
    root[0] = 0
    labm = root[comp]
    u, inv = np.unique(labm, return_inverse=True)
    labm = inv.reshape(labm.shape).astype(np.int32)
    rles = encode_masks(labm)
    tps, fp, fn = 0.0, 0, 0
    ntp = 0
    for gt in gts_by[s]:
        a = PQAccumulator()
        a.add(gt, rles)
        tps += sum(a.tp_iou); ntp += a.tp; fp += a.fp; fn += a.fn
    _CACHE[key] = (tps, ntp, fp, fn)
    return _CACHE[key]


def score(stem_list, data, gts_by, joins_by):
    t = np.zeros(4)
    for s in stem_list:
        t += _img_stats(s, data, gts_by, joins_by[s])
    return t[0] / (t[1] + 0.5 * t[2] + 0.5 * t[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--v2", action="store_true")
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    items = load_train()
    groups = by_image(items)
    stems = sorted(p.stem for p in MAPS.glob("*.npy"))
    if a.n:
        stems = stems[: a.n]
    fold_of = {}
    for f in range(5):
        _, va = fold_split(items, f)
        for t in va:
            fold_of[t.image_id] = f
    work = [(s, [t.gt_rles for t in groups[s]], a.v2) for s in stems]
    data, X, P = {}, {}, {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for stem, comp, rows, pairs, grp in ex.map(feats_image, work, chunksize=2):
            data[stem] = (comp, pairs, grp)
            X[stem], P[stem] = rows, pairs
    gts_by = {s: [t.gt_rles for t in groups[s]] for s in stems}
    # OOF join probabilities
    prob_by = {}
    for f in range(5):
        tr = [s for s in stems if fold_of[s] != f]
        te = [s for s in stems if fold_of[s] == f]
        Xtr = np.concatenate([X[s] for s in tr if len(X[s])])
        ytr = np.concatenate([np.array([p[2] for p in P[s]]) for s in tr if len(P[s])])
        ok = ytr >= 0
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                             random_state=0).fit(Xtr[ok], ytr[ok])
        for s in te:
            prob_by[s] = clf.predict_proba(X[s])[:, 1] if len(X[s]) else np.zeros(0)
    yall = np.concatenate([np.array([p[2] for p in P[s]]) for s in stems if len(P[s])])
    pall = np.concatenate([prob_by[s] for s in stems if len(P[s])])
    rad = np.concatenate([X[s][:, 14] for s in stems if len(X[s])])
    ok = yall >= 0
    print(f"images {len(stems)}; candidate pairs {len(yall)}; labelled {ok.sum()} (same {int((yall == 1).sum())}, "
          f"different {int((yall == 0).sum())}); radius rule joins {int(rad.sum())}")
    print(f"AUC learned {roc_auc_score(yall[ok], pall[ok]):.3f}; radius-rule accuracy on labelled "
          f"{((rad[ok] > 0.5) == (yall[ok] == 1)).mean():.3f}; learned@0.5 {((pall[ok] > 0.5) == (yall[ok] == 1)).mean():.3f}")
    radius_joins = {s: [bool(x[14]) for x in X[s]] for s in stems}
    base = score(stems, data, gts_by, radius_joins)
    ts = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    per_t = {t: {s: list(prob_by[s] > t) for s in stems} for t in ts}
    rng = np.random.default_rng(0)
    # cross-tune the threshold
    deltas = []
    picks = []
    imgs_scores = {}
    for rep in range(3):
        perm = list(rng.permutation(stems))
        A, B = perm[: len(perm) // 2], perm[len(perm) // 2:]
        for h1, h2 in ((A, B), (B, A)):
            best = max(ts, key=lambda t: score(h1, data, gts_by, per_t[t]))
            picks.append(best)
            deltas.append(score(h2, data, gts_by, per_t[best]) - score(h2, data, gts_by, radius_joins))
    print(f"radius PQ {base:.4f}")
    for t in ts:
        print(f"  learned t={t}: PQ {score(stems, data, gts_by, per_t[t]):.4f} (in-sample threshold)")
    print(f"cross-tuned delta vs radius: mean {np.mean(deltas):+.4f} (halves: {np.round(deltas, 4).tolist()}), picks {picks}")
    # paired bootstrap at the most picked threshold
    t = max(set(picks), key=picks.count)
    rs = []
    for _ in range(1000):
        b = list(rng.choice(stems, len(stems)))
        rs.append(score(b, data, gts_by, per_t[t]) - score(b, data, gts_by, radius_joins))
    print(f"t={t}: delta {score(stems, data, gts_by, per_t[t]) - base:+.4f}  bootstrap 95% "
          f"[{np.percentile(rs, 2.5):+.4f}, {np.percentile(rs, 97.5):+.4f}]")
    fold_d = []
    for f in range(5):
        fs = [x for x in stems if fold_of[x] == f]
        fold_d.append(score(fs, data, gts_by, per_t[t]) - score(fs, data, gts_by, radius_joins))
    print(f"t={t}: per-fold delta {np.round(fold_d, 4).tolist()} ({sum(d > 0 for d in fold_d)}/5 positive)")


if __name__ == "__main__":
    main()
