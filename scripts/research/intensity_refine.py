"""Intensity-guided boundary refinement of predicted instances (research 05/11, literature idea 3).

The filament literature (Diercke 2024 pseudo-masks `I <= mean - std` in a box; MORDEN DenseCRF) snaps
mask edges to the dark H-alpha pixels. Test on the N1+N2 OOF maps: instances from the final-1 hysteresis
config, then for each instance, local background stats in its bbox + WIN px (pixels outside every
instance and on the disk; median and 1.4826*MAD), z = (I - median) / std, and
  - add:    unclaimed pixels within r px outside the instance with z < -b
  - remove: pixels within r px inside the instance edge with z > -a (not darker than a sigma)
The combo (none / add / remove / both) is cross-tuned: 3 random halvings, chosen on half A, scored on
half B, both ways; then a paired bootstrap (1000 resamples by image) at the most-picked combo, and the
per-fold deltas. Acceptance (cheap change, F48): 95% CI > 0 on the 707 OOF images.
Usage: .venv/bin/python scripts/research/intensity_refine.py [--folds 0] [--jobs 6]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filseg.data import by_image, encode_masks, fold_split, load_gray, load_train  # noqa: E402
from filseg.dl import PostParams, instances_lowres, upsample_labels  # noqa: E402
from filseg.metric import PQAccumulator  # noqa: E402

MAPS = ROOT / "outputs" / "ens" / "n1n2_oof"
CFG = json.loads((ROOT / "configs" / "post_ens_n1n2_hyst.json").read_text())
WIN = 48
DISK_MIN = 20   # uint8 intensity below this = off-disk sky
COMBOS = ([None]
          + [("add", b, r) for b in (0.5, 1.0, 1.5) for r in (2, 4)]
          + [("rm", a, 2) for a in (0.0, 0.5, 1.0)]
          + [("both", (0.5, 1.0), 2), ("both", (0.5, 1.0), 4), ("both", (0.0, 1.0), 2)])


def _disk(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))


def refine(inst, img, combo):
    if combo is None:
        return inst
    kind, par, r = combo
    a, b = (par if kind == "both" else (par, par))
    out = inst.copy()
    H, W = inst.shape
    on_disk = img > DISK_MIN
    n = int(inst.max())
    if n == 0:
        return inst
    ys, xs = np.nonzero(inst)
    lab = inst[ys, xs]
    y0 = np.full(n + 1, H); y1 = np.zeros(n + 1, int); x0 = np.full(n + 1, W); x1 = np.zeros(n + 1, int)
    np.minimum.at(y0, lab, ys); np.maximum.at(y1, lab, ys); np.minimum.at(x0, lab, xs); np.maximum.at(x1, lab, xs)
    k = _disk(r)
    for i in range(1, n + 1):
        ya, yb = max(0, y0[i] - WIN), min(H, y1[i] + WIN + 1)
        xa, xb = max(0, x0[i] - WIN), min(W, x1[i] + WIN + 1)
        sub = inst[ya:yb, xa:xb]
        im = img[ya:yb, xa:xb].astype(np.float32)
        bg = (sub == 0) & on_disk[ya:yb, xa:xb]
        if bg.sum() < 50:
            continue
        v = im[bg]
        med = float(np.median(v))
        sd = max(1.4826 * float(np.median(np.abs(v - med))), 1.0)
        z = (im - med) / sd
        me = (sub == i).astype(np.uint8)
        o = out[ya:yb, xa:xb]
        if kind in ("add", "both"):
            ring = (cv2.dilate(me, k) > 0) & (sub == 0) & (o == 0)
            o[ring & (z < -b)] = i
        if kind in ("rm", "both"):
            edge = (me > 0) & (cv2.erode(me, k) == 0)
            o[edge & (z > -a)] = 0
    return out


def image_job(args):
    stem, path, gts = args
    prob = np.load(MAPS / f"{stem}.npy").astype(np.float32) / 255.0
    inst = upsample_labels(instances_lowres(prob, PostParams(**CFG)))
    img = load_gray(path)
    rows = []
    for c in COMBOS:
        rles = encode_masks(refine(inst, img, c))
        t = np.zeros(4)
        for gt in gts:
            acc = PQAccumulator()
            acc.add(gt, rles)
            t += (sum(acc.tp_iou), acc.tp, acc.fp, acc.fn)
        rows.append(t)
    return stem, np.array(rows)


def pq(s):
    return s[0] / max(s[1] + 0.5 * s[2] + 0.5 * s[3], 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    items = load_train()
    groups = by_image(items)
    fold_of = {}
    for f in range(5):
        for t in fold_split(items, f)[1]:
            fold_of[t.image_id] = f
    stems = sorted(s for s in groups if fold_of.get(s) in a.folds and (MAPS / f"{s}.npy").exists())
    work = [(s, groups[s][0].path, [t.gt_rles for t in groups[s]]) for s in stems]
    with ProcessPoolExecutor(a.jobs) as ex:
        res = dict(ex.map(image_job, work, chunksize=2))
    S = np.stack([res[s] for s in stems])          # (n_img, n_combo, 4)
    tot = S.sum(0)
    print(f"{len(stems)} images, folds {a.folds}, maps {MAPS.name}, cfg {CFG}")
    for c, t in zip(COMBOS, tot):
        print(f"  {str(c):28s} PQ {pq(t):.4f}  TP {t[1]:.0f} FP {t[2]:.0f} FN {t[3]:.0f}  d {pq(t) - pq(tot[0]):+.4f}")
    rng = np.random.default_rng(0)
    deltas, picks = [], []
    for _ in range(3):
        perm = rng.permutation(len(stems))
        A, B = perm[: len(perm) // 2], perm[len(perm) // 2:]
        for h1, h2 in ((A, B), (B, A)):
            sa, sb = S[h1].sum(0), S[h2].sum(0)
            best = int(np.argmax([pq(x) for x in sa]))
            picks.append(best)
            deltas.append(pq(sb[best]) - pq(sb[0]))
    print(f"cross-tuned delta: mean {np.mean(deltas):+.4f} halves {np.round(deltas, 4).tolist()} "
          f"picks {[str(COMBOS[p]) for p in picks]}")
    best = max(set(picks), key=picks.count)
    bs = []
    for _ in range(1000):
        idx = rng.integers(0, len(stems), len(stems))
        t = S[idx].sum(0)
        bs.append(pq(t[best]) - pq(t[0]))
    print(f"most picked {COMBOS[best]}: delta {pq(tot[best]) - pq(tot[0]):+.4f} bootstrap 95% "
          f"[{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
    fd = []
    for f in a.folds:
        idx = [k for k, s in enumerate(stems) if fold_of[s] == f]
        t = S[idx].sum(0)
        fd.append(round(pq(t[best]) - pq(t[0]), 4))
    print(f"per-fold delta {fd}")


if __name__ == "__main__":
    main()
