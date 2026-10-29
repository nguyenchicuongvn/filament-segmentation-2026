"""Select a non-overlapping set of component-tree nodes per image.

oracle  : node value = true matched IoU (Dinkelbach on lambda = PQ) -> ceiling of any selection
learned : GBM predicts mean matched IoU per annotator entry; 5-fold CV by image; lambda picked on
          the training folds; scored on the held-out fold (honest).
base    : level 0.5 nodes with pmax > 0.99 and area >= 50 (approximates the hysteresis post)
Paired bootstrap over images against the records.pkl baseline (the real E9a hyst pipeline).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

S = Path(__file__).parent
TREE = Path(sys.argv[1]) if len(sys.argv) > 1 else S / "tree_unet1024_effb4_r4.pkl"
data = pickle.load(open(TREE, "rb"))
rec = pickle.load(open(S / "records.pkl", "rb"))

# baseline per image from the real pipeline
base_img = {}
for e in rec:
    n = sum(e["tp_iou"]); tp = len(e["tp_iou"])
    fp = sum(not p["tp"] for p in e["preds"]); fn = sum(not g["tp"] for g in e["gts"])
    a = base_img.setdefault(e["stem"], [0.0, 0.0])
    a[0] += n; a[1] += tp + .5 * fp + .5 * fn

FEAT = ["level", "log_area", "pmean", "pmax", "p25", "p75", "frac90", "elong", "major", "extent",
        "contrast", "flat", "nfrag", "r", "absx", "absy", "par_ratio", "nchild", "child_ratio", "depth"]
rows = []
for im in data:
    c = im["cands"]
    ne = len(im["ngt"])
    ch = [[] for _ in c]
    for i, x in enumerate(c):
        if x["parent"] >= 0:
            ch[x["parent"]].append(i)
    im["children"] = ch
    for i, x in enumerate(c):
        ious = np.array(x["ious"])
        m = np.where(ious > 0.5, ious, 0.0)
        par = c[x["parent"]] if x["parent"] >= 0 else None
        d = 0; p = x["parent"]
        while p >= 0:
            d += 1; p = c[p]["parent"]
        rows.append(dict(stem=im["stem"], idx=i, n_e=ne, y=m.mean(), msum=m.sum(), ntp=(ious > 0.5).sum(),
                         par_ratio=x["area"] / par["area"] if par else 1.0, nchild=len(ch[i]),
                         child_ratio=max([c[j]["area"] for j in ch[i]], default=0) / x["area"], depth=d,
                         **{k: x[k] for k in FEAT if k in x}))
D = pd.DataFrame(rows)
stems = sorted({im["stem"] for im in data})
print("images", len(stems), "nodes", len(D))


def select(im, score):
    """score: per-node gain array (already value - cost). returns selected node idx list."""
    c = im["cands"]; ch = im["children"]
    order = sorted(range(len(c)), key=lambda i: -c[i]["li"])  # deepest first
    best = np.zeros(len(c)); take = [None] * len(c)
    for i in order:
        s_ch = sum(best[j] for j in ch[i])
        if score[i] >= s_ch and score[i] > 0:
            best[i] = score[i]; take[i] = "self"
        else:
            best[i] = max(s_ch, 0.0); take[i] = "kids"
    sel = []

    def walk(i):
        if take[i] == "self":
            sel.append(i)
        else:
            for j in ch[i]:
                walk(j)
    for i, x in enumerate(c):
        if x["parent"] < 0:
            walk(i)
    return sel


def evaluate(sel_by_stem):
    out = {}
    for im in data:
        sel = sel_by_stem[im["stem"]]
        c = im["cands"]; ne = len(im["ngt"])
        num = 0.0; tp = 0
        for i in sel:
            ious = np.array(c[i]["ious"]); h = ious > 0.5
            num += ious[h].sum(); tp += h.sum()
        den = 0.5 * (len(sel) * ne + sum(im["ngt"]))
        out[im["stem"]] = (num, den)
    return out


def pq(d, stems_=None):
    ks = stems_ or list(d)
    return sum(d[k][0] for k in ks) / sum(d[k][1] for k in ks)


def boot(a, b, n=2000, seed=0):
    ks = [k for k in a if k in b]
    A = np.array([a[k] for k in ks]); B = np.array([b[k] for k in ks])
    rng = np.random.default_rng(seed); ds = []
    for _ in range(n):
        ix = rng.integers(0, len(ks), len(ks))
        ds.append(B[ix, 0].sum() / B[ix, 1].sum() - A[ix, 0].sum() / A[ix, 1].sum())
    ds = np.array(ds)
    return B[:, 0].sum() / B[:, 1].sum() - A[:, 0].sum() / A[:, 1].sum(), np.percentile(ds, 2.5), np.percentile(ds, 97.5)


print("baseline (real hyst pipeline) PQ", round(pq(base_img), 4))
by = {im["stem"]: im for im in data}
Dg = {s: g for s, g in D.groupby("stem")}
for _s in by:
    Dg.setdefault(_s, D.iloc[0:0])

# base in tree
sel = {}
for im in data:
    c = im["cands"]
    sel[im["stem"]] = [i for i, x in enumerate(c) if x["level"] == 0.5 and x["pmax"] > 0.99 and x["area"] >= 50]
r = evaluate(sel); print("base-in-tree PQ", round(pq(r), 4), "vs real", [round(v, 4) for v in boot(base_img, r)])

# oracle (Dinkelbach)
lam = 0.41
for it in range(6):
    sel = {}
    for im in data:
        g = Dg[im["stem"]].sort_values("idx")
        sel[im["stem"]] = select(im, (g.msum - lam * 0.5 * g.n_e).values)
    r = evaluate(sel); lam = pq(r)
print("ORACLE selection PQ", round(lam, 4), "(ceiling for any per-node choice on this tree)")
# oracle restricted: only level choice for existing base instances? skip

# single fixed level (global threshold, no seed): sanity
for lv in (0.4, 0.5, 0.6):
    sel = {im["stem"]: [i for i, x in enumerate(im["cands"]) if x["level"] == lv and x["area"] >= 50] for im in data}
    print(f"all nodes at level {lv}: PQ {pq(evaluate(sel)):.4f}")

# learned, 5-fold CV by image
rng = np.random.default_rng(42)
perm = list(rng.permutation(stems))
folds = [set(perm[i::5]) for i in range(5)]
LAMS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
for rep_seed in (0,):
    sel_cv = {}
    chosen = []
    for f in folds:
        tr = D[~D.stem.isin(f)]; te = D[D.stem.isin(f)]
        mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                            min_samples_leaf=40, l2_regularization=1.0, random_state=0)
        mdl.fit(tr[FEAT], tr.y)
        # pick lambda on training folds with inner 4-fold OOF predictions
        inner = [s for s in stems if s not in f]
        ifolds = [set(inner[i::4]) for i in range(4)]
        oof = pd.Series(0.0, index=tr.index)
        for g2 in ifolds:
            a = tr[~tr.stem.isin(g2)]; b = tr[tr.stem.isin(g2)]
            m2 = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                               min_samples_leaf=40, l2_regularization=1.0, random_state=0)
            m2.fit(a[FEAT], a.y)
            oof[b.index] = m2.predict(b[FEAT])
        best = None
        trs = {st: oof[g.sort_values("idx").index].values for st, g in tr.groupby("stem")}
        for lm in LAMS:
            s2 = {}
            for st in inner:
                s2[st] = select(by[st], trs[st] - lm * 0.5) if st in trs else []
            sub = {k: v for k, v in s2.items()}
            ev = evaluate({**{k: [] for k in by}, **sub})
            val = pq(ev, inner)
            if best is None or val > best[0]:
                best = (val, lm)
        chosen.append(best[1])
        pred = pd.Series(mdl.predict(te[FEAT]), index=te.index)
        for st in f:
            sel_cv[st] = []
        for st, g in te.groupby("stem"):
            sel_cv[st] = select(by[st], (pred[g.sort_values("idx").index] - best[1] * 0.5).values)
    r = evaluate(sel_cv)
    d, lo, hi = boot(base_img, r)
    print(f"LEARNED tree selection (CV by image) PQ {pq(r):.4f}  vs real base: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  lambdas {chosen}")
    pickle.dump(r, open(S / f"learned_{TREE.stem}.pkl", "wb"))
    # what changed: count + level histogram
    lv = [by[s]["cands"][i]["level"] for s in sel_cv for i in sel_cv[s]]
    print("selected nodes", len(lv), "level histogram", pd.Series(lv).value_counts().sort_index().to_dict())
