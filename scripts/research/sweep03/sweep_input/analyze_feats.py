"""Analyse feats.pkl: (1) PQ per station; (2) do image filters (Sato, black-hat, DoG, contrast)
separate good predictions (mean matched IoU > PQ/2, RESEARCH F26) from bad ones, alone and on top
of map features (5-fold-by-image CV AUC, HistGradientBoosting); (3) are missed GTs fainter than
detected ones, split by support from other annotators."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

res = pickle.load(open(Path(__file__).parent / "feats.pkl", "rb"))
ent = pd.DataFrame([e for r in res for e in r["ent"]])
P = pd.DataFrame([p for r in res for p in r["preds"]])
G = pd.DataFrame([g for r in res for g in r["gts"]])
img = pd.DataFrame([dict(stem=r["stem"], st=r["station"], **r["img"]) for r in res])


def pq(d):
    return d.siou.sum() / (d.tp.sum() + 0.5 * d.fp.sum() + 0.5 * d.fn.sum())


tot = pq(ent)
print(f"overall PQ {tot:.4f}  TP {ent.tp.sum()} FP {ent.fp.sum()} FN {ent.fn.sum()}")
ent["grp"] = ent.key.str[:2]
rng = np.random.default_rng(0)
print("\nPQ per station (bootstrap SD over images):")
for s, d in ent.groupby("station"):
    stems = d.stem.unique()
    bs = [pq(pd.concat([d[d.stem == x] for x in rng.choice(stems, len(stems))])) for _ in range(200)]
    gshare = (d.grp.isin(["01", "04"])).mean()
    print(f"  {s}: n_img {len(stems):3d} entries {len(d):3d}  PQ {pq(d):.3f} ± {np.std(bs):.3f}  share grp01/04 {gshare:.2f}")
# station effect after annotator group: PQ per station within group 01 only and non-01
for lab, m in [("grp01/04", ent.grp.isin(["01", "04"])), ("other grps", ~ent.grp.isin(["01", "04"]))]:
    print(f"  {lab}: " + "  ".join(f"{s} {pq(d):.3f}" for s, d in ent[m].groupby("station")))
print("\nimage stats per station (holdout):")
print(img.groupby("st")[["raw_med", "limb_ratio", "flat_std", "lap_var", "hp_std", "sat_mean"]].median().round(4))
# per-image PQ vs image quality: correlation
pi = ent.groupby("stem").apply(lambda d: pd.Series(dict(pq=pq(d), n=len(d)))).join(img.set_index("stem"))
for c in ["raw_med", "limb_ratio", "flat_std", "lap_var", "hp_std", "disk_r"]:
    r = pi[["pq", c]].corr(method="spearman").iloc[0, 1]
    print(f"  spearman(per-image PQ, {c}) = {r:+.3f}")
q = pi.lap_var.quantile([1 / 3, 2 / 3]).values
for lab, m in [("sharpness low", pi.lap_var <= q[0]), ("mid", (pi.lap_var > q[0]) & (pi.lap_var <= q[1])), ("high", pi.lap_var > q[1])]:
    print(f"  {lab:14s} PQ {pq(ent[ent.stem.isin(pi.index[m])]):.3f}")

# (2) instance separability
thr = tot / 2
P["good"] = P.miou > thr
print(f"\npreds {len(P)}, good (mIoU > {thr:.3f}) {P.good.mean():.3f}, anyTP {P.anytp.mean():.3f}")
filt = ["flat", "flatmin", "contrast", "bh", "bhmax", "dog", "sat", "satmax", "sat_ratio"]
base = ["pmean", "pmax", "p90", "area", "elong", "r"]
print("single-feature AUC (good vs bad), oriented >0.5:")
for c in base + filt:
    a = roc_auc_score(P.good, P[c])
    print(f"  {c:10s} {max(a, 1 - a):.3f} ({'+' if a >= .5 else '-'})")


def cv_auc(cols, y, groups, seeds=3):
    out = []
    for sd in range(seeds):
        uniq = np.unique(groups)
        perm = {g: i for i, g in enumerate(np.random.default_rng(sd).permutation(uniq))}
        gk = np.array([perm[g] for g in groups])
        pr = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(P, y, gk):
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=sd)
            m.fit(P.iloc[tr][cols], y[tr])
            pr[te] = m.predict_proba(P.iloc[te][cols])[:, 1]
        out.append(roc_auc_score(y, pr))
    return np.mean(out), np.std(out)


y = P.good.values
for lab, cols in [("map only", base), ("filters only", filt + ["area", "r"]), ("map + filters", base + filt)]:
    m, s = cv_auc(cols, y, P.stem.values)
    print(f"CV AUC {lab:14s} {m:.4f} ± {s:.4f}")
# among confident preds only (pmax >= 0.999): the regime F34 says is the problem
conf = P.pmax >= 0.995
print(f"confident preds (pmax>=0.995): {conf.sum()}, good {P.good[conf].mean():.3f}")
for c in filt:
    a = roc_auc_score(P.good[conf], P[c][conf])
    print(f"  {c:10s} AUC among confident {max(a, 1 - a):.3f}")

# (3) GT faintness: detected vs missed, by support
G["small"] = G.area2048 < 800
print("\nGT image contrast (flat-field in/out ratio; lower = darker) and Sato, median:")
for (multi, sup), d in G.assign(multi=G.nann > 1).groupby(["multi", G.sup > 0]):
    for det, dd in d.groupby("det"):
        print(f"  multi={multi!s:5} supported={sup!s:5} det={det!s:5} n={len(dd):4d} contrast {dd.contrast.median():.3f} "
              f"bh {dd.bh.median():.4f} sat {dd.sat.median():.4f} area {dd.area2048.median():.0f}")
# within-size control: AUC of contrast for det among GTs (area-matched by bins)
for lo, hi in [(0, 800), (800, 3200), (3200, 1e9)]:
    d = G[(G.area2048 >= lo) & (G.area2048 < hi)]
    a = roc_auc_score(d.det, -d.contrast)
    a2 = roc_auc_score(d.det, d.sat)
    print(f"  area [{lo},{hi}) n={len(d)} det rate {d.det.mean():.2f}  AUC(det | darker contrast) {a:.3f}  AUC(det | sato) {a2:.3f}")
# GT drawn by only one annotator in multi-annotator images vs supported: contrast differs?
d = G[G.nann > 1]
a = roc_auc_score(d.sup > 0, -d.contrast)
print(f"  multi-ann GTs: AUC(supported | darker) {a:.3f}; median contrast supported {d[d.sup>0].contrast.median():.3f} vs lone {d[d.sup==0].contrast.median():.3f}")
pickle.dump(dict(P=P, G=G, ent=ent, img=img), open(Path(__file__).parent / "tables.pkl", "wb"))
