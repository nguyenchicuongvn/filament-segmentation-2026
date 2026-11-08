"""Ceiling of any component-grouping step (option b) on the N1+N2 OOF ensemble maps (2048 px).

Components = kept hysteresis components (thresh/seed/min_area of configs/post_ens_n1n2_hyst.json),
grouped by:
  none     no grouping (radius 0)
  radius   current rule (dilate by group_radius, configs value 12 at 2048)
  oracle   start from 'radius', then cut every merge whose components belong to different GT filaments
           and join every pair of components belonging to the same GT filament ("consensus GT":
           a component's GT in an annotator entry = the GT covering >= 50% of its pixels; two
           components are 'same' if they share a GT in >= half the entries that assign both,
           'different' if they get different GTs in >= half). Components without a GT keep 'radius'.
  oracle_join   only the joins of 'oracle' (cuts disabled)
  oracle_cut    only the cuts of 'oracle'
The oracle uses labels the model cannot see, so it is an upper bound for a learned grouping model.
Usage: .venv/bin/python scripts/research/spine_group_oracle.py [--n 707] [--jobs 8]
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
from filseg.data import by_image, encode_masks, load_train  # noqa: E402
from filseg.metric import PQAccumulator  # noqa: E402

MAPS = ROOT / "outputs" / "ens" / "n1n2_oof"
CFG = json.loads((ROOT / "configs" / "post_ens_n1n2_hyst.json").read_text())
MODES = ["none", "radius", "oracle", "oracle_join", "oracle_cut"]


def disk(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def f(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def u(self, a, b):
        self.p[self.f(a)] = self.f(b)


def one(args):
    stem, gts = args  # gts: list (per entry) of list of RLEs
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
    # radius grouping: which components are joined
    r = CFG["group_radius"]
    grown = cv2.dilate((comp > 0).astype(np.uint8), disk(r))
    _, glab = cv2.connectedComponents(grown, connectivity=8)
    area = np.bincount(comp.ravel(), minlength=m + 1)
    grp = np.zeros(m + 1, np.int64)
    pix = comp > 0
    grp[comp[pix]] = glab[pix]
    # GT assignment per entry
    assign = np.full((len(gts), m + 1), -1, np.int64)
    for e, rles in enumerate(gts):
        if not rles:
            continue
        g = mu.decode(rles)  # H W K
        for k in range(g.shape[2]):
            ov = np.bincount(comp[g[:, :, k] > 0], minlength=m + 1)
            hit = ov >= 0.5 * area
            hit[0] = False
            assign[e, hit] = k
    def rel(i, j):
        both = (assign[:, i] >= 0) & (assign[:, j] >= 0)
        nb = both.sum()
        if nb == 0:
            return 0
        same = (assign[both, i] == assign[both, j]).sum()
        if same * 2 >= nb:
            return 1
        return -1
    out = {}
    out["none"] = comp
    out["radius"] = np.where(pix, glab, 0)
    for mode in ("oracle", "oracle_join", "oracle_cut"):
        uf = UF(m + 1)
        for i in range(1, m + 1):
            for j in range(i + 1, m + 1):
                rr = rel(i, j)
                radius_join = grp[i] == grp[j]
                join = radius_join
                if rr == 1 and mode in ("oracle", "oracle_join"):
                    join = True
                if rr == -1 and radius_join and mode in ("oracle", "oracle_cut"):
                    join = False
                if join:
                    uf.u(i, j)
        root = np.array([uf.f(i) for i in range(m + 1)])
        root[0] = 0
        out[mode] = root[comp]
    res = {}
    for mode in MODES:
        lab_m = out[mode]
        u, inv = np.unique(lab_m, return_inverse=True)
        lab_m = inv.reshape(lab_m.shape).astype(np.int32)
        if u[0] != 0:
            lab_m += 1
        rles = encode_masks(lab_m)
        accs = []
        for gt in gts:
            a = PQAccumulator()
            a.add(gt, rles)
            accs.append((a.tp_iou, a.fp, a.fn, a.gt_split, a.pred_merge))
        res[mode] = accs
    return stem, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    groups = by_image(load_train())
    stems = sorted(p.stem for p in MAPS.glob("*.npy"))
    if a.n:
        stems = stems[: a.n]
    work = [(s, [t.gt_rles for t in groups[s]]) for s in stems]
    per_img = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for stem, res in ex.map(one, work, chunksize=2):
            per_img[stem] = res
    def pq(stems_, mode):
        tp, fp, fn, sp, mg = [], 0, 0, 0, 0
        for s in stems_:
            for t, f1, f2, g1, g2 in per_img[s][mode]:
                tp += t; fp += f1; fn += f2; sp += g1; mg += g2
        den = len(tp) + 0.5 * fp + 0.5 * fn
        return sum(tp) / den, len(tp), fp, fn, sp, mg
    rng = np.random.default_rng(0)
    base = pq(stems, "radius")[0]
    print(f"images {len(stems)}; config {CFG}")
    for mode in MODES:
        v, tp, fp, fn, sp, mg = pq(stems, mode)
        ds = []
        for _ in range(500):
            b = list(rng.choice(stems, len(stems)))
            ds.append(pq(b, mode)[0] - pq(b, "radius")[0])
        lo, hi = np.percentile(ds, [2.5, 97.5])
        print(f"{mode:12s} PQ {v:.4f}  d_vs_radius {v - base:+.4f} [{lo:+.4f}, {hi:+.4f}]  TP {tp} FP {fp} FN {fn} "
              f"gt_split {sp} pred_merge {mg}")


if __name__ == "__main__":
    main()
