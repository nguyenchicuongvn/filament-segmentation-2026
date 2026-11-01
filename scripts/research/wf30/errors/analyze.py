"""Bucket the FPs/FNs of records.pkl and compute PQ-if-fixed counterfactuals."""
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

S = Path(__file__).parent
rec = pickle.load(open(S / "records.pkl", "rb"))
P = pd.DataFrame([p | dict(nann=e["nann"]) for e in rec for p in e["preds"]])
G = pd.DataFrame([g for e in rec for g in e["gts"]]).rename(columns={"cov": "gcov"})
for d in (P, G):
    d["grp"] = d.key.str[:2]
    d["alon"] = d.lon.abs()

S0 = sum(sum(e["tp_iou"]) for e in rec)
FP0 = int((~P.tp).sum()); FN0 = int((~G.tp).sum())
TP0 = int(G.tp.sum())  # one GT-overlap case: a pred matches 2 overlapping GTs


def pq(s, tp, fp, fn):
    return s / (tp + .5 * fp + .5 * fn)


PQ0 = pq(S0, TP0, FP0, FN0)
print(f"BASE PQ {PQ0:.4f} SQ {S0/TP0:.3f} TP {TP0} FP {FP0} FN {FN0}  entries {len(rec)} images {P.stem.nunique()}")


# ---------------- FP buckets
def fp_bucket(r):
    if r.tp:
        return "TP"
    if r.other:
        return "other_annot"          # matches another annotator's GT of this image
    if r.ncov >= 2:
        return "merge"
    if 0.3 <= r.best <= 0.5:
        return "near_fat" if r.ratio > 1 else "near_thin"
    if r.in_gt >= 0.5:
        return "fragment"             # mostly inside one GT but IoU < 0.3
    if r.best >= 0.1:
        return "partial"
    # orphans by location
    if r.alon > 70:
        return "orphan_lon>70"
    if r.r > 0.92:
        return "orphan_limb"
    if r.sun_d < 0.04:
        return "orphan_sunspot"
    return "orphan_disk"


P["b"] = P.apply(fp_bucket, axis=1)
fp = P[~P.tp]
print("\n== FP buckets (per annotator entry): count, share, PQ if removed (oracle)")
for b, n in fp.b.value_counts().items():
    print(f"  {b:16s} {n:5d} {n/FP0:5.1%}  PQ->{pq(S0, TP0, FP0-n, FN0):.4f} (+{pq(S0, TP0, FP0-n, FN0)-PQ0:.4f})")
print(f"  ALL FP removed          PQ->{pq(S0, TP0, 0, FN0):.4f}")

# orphan FPs by nann (single-annotator images)
print("\n== FP rate by #annotators: FP/TP")
for k, d in P.groupby("nann"):
    print(f"  nann={k}: preds {len(d)} TP {d.tp.sum()} FP {(~d.tp).sum()} FP/TP {(~d.tp).sum()/max(d.tp.sum(),1):.2f}  orphan share {d.b.str.startswith('orphan').sum()/max((~d.tp).sum(),1):.2f}")


# ---------------- FN buckets
def fn_bucket(r):
    if r.tp:
        return "TP"
    if r.ntouch >= 2 and r.union_iou > 0.5:
        return "split"
    if r.merged:
        return "merged"
    if 0.3 <= r.best <= 0.5:
        return "near_fat" if r.ratio > 1 else "near_thin"
    if r.gcov < 0.1 and r.best < 0.1:
        base = "missed"
        if r.alon > 70:
            return base + "_lon>70"
        return base + ("_solo" if r.sup == 0 and r.nann > 1 else ("_1ann" if r.nann == 1 else "_supported"))
    return "partial"


G["b"] = G.apply(fn_bucket, axis=1)
fn = G[~G.tp]
print("\n== FN buckets: count, share; median area (2048 px), share with sup==0 (multi-annot imgs)")
for b, n in fn.b.value_counts().items():
    d = fn[fn.b == b]
    print(f"  {b:18s} {n:5d} {n/FN0:5.1%}  med_area {d.area.median():7.0f}  alon>70 {np.mean(d.alon>70):.2f}  r>0.92 {np.mean(d.r>0.92):.2f}")

# ---------------- counterfactual fixes
pk = P.set_index(["key", "j"])
# near miss fix: GT with best in [0.3,0.5] whose best pred is FP in this entry -> TP @0.55 (and at its own IoU+0.15)
near = fn[fn.b.str.startswith("near")]
nfp = sum(not pk.loc[(r.key, r.bj)].tp for r in near.itertuples())
for b in ("near_fat", "near_thin"):
    d = near[near.b == b]
    n = len(d); nf = sum(not pk.loc[(r.key, r.bj)].tp for r in d.itertuples())
    print(f"\nfix {b}: {n} GTs -> TP@0.55: PQ {pq(S0+.55*n, TP0+n, FP0-nf, FN0-n):.4f} (+{pq(S0+.55*n, TP0+n, FP0-nf, FN0-n)-PQ0:.4f})")
n = len(near)
print(f"fix all near: +{pq(S0+.55*n, TP0+n, FP0-nfp, FN0-n)-PQ0:.4f}")

# split fix: union of touching preds
sp = fn[fn.b == "split"]
dS = dTP = dFP = dFN = 0
for r in sp.itertuples():
    dS += r.union_iou; dTP += 1; dFN -= 1
    dFP -= sum(not pk.loc[(r.key, j)].tp for j in r.touching)
print(f"fix splits ({len(sp)}): +{pq(S0+dS, TP0+dTP, FP0+dFP, FN0+dFN)-PQ0:.4f}")

# merge fix: proportional partition of the merged pred
mg = G[G.merged & (G.gcov > 0)]
dS = dTP = dFP = dFN = 0
for (key, j), d in mg.groupby(["key", "bj"]):
    pr = pk.loc[(key, j)]
    if pr.ncov < 2:
        continue
    I = d.gcov * d.area
    est = I / (d.area + I / max(pr.in_gt, 1e-6) - I)
    old = d.tp.values
    new = est.values > 0.5
    dS += est.values[new].sum() - d.best.values[old].sum()
    dTP += new.sum() - old.sum()
    dFN -= new.sum() - old.sum()
    # pred itself was FP if none old matched; after partition parts unmatched become FP
    dFP += (len(d) - new.sum()) - (0 if old.any() else 1)
print(f"fix merges (pairs {len(mg)}): +{pq(S0+dS, TP0+dTP, FP0+dFP, FN0+dFN)-PQ0:.4f}  dTP {dTP} dFP {dFP}")

# SQ ceiling
for target in (0.70, 0.75):
    ious = np.concatenate([e["tp_iou"] for e in rec])
    s2 = np.maximum(ious, target).sum()
    print(f"raise every TP IoU to >= {target}: +{pq(s2, TP0, FP0, FN0)-PQ0:.4f}")

# ---------------- by annotator group
print("\n== by annotator group: PQ, FP/TP, FN/TP, top FP buckets share, top FN buckets share")
ent = pd.DataFrame([dict(key=e["key"], s=sum(e["tp_iou"])) for e in rec])
ent["grp"] = ent.key.str[:2]
for g, d in ent.groupby("grp"):
    p = P[P.grp == g]; q = G[G.grp == g]
    tp, f1, f2 = p.tp.sum(), (~p.tp).sum(), (~q.tp).sum()
    fb = p[~p.tp].b.value_counts(normalize=True).round(2).head(4).to_dict()
    nb = q[~q.tp].b.value_counts(normalize=True).round(2).head(4).to_dict()
    print(f"  {g}: n_ent {len(d)} PQ {pq(d.s.sum(), tp, f1, f2):.3f} FP/TP {f1/tp:.2f} FN/TP {f2/tp:.2f} gt/ent {len(q)/len(d):.1f} medGTarea {q.area.median():.0f}\n     FP {fb}\n     FN {nb}")

# ---------------- descriptor contrasts
print("\n== descriptors: TP preds vs orphan FPs vs other FPs (medians)")
cols = ["area", "elong", "contrast", "flatmin", "pmean", "pmax", "r", "alon", "sun_d"]
P["kind"] = np.where(P.tp, "TP", np.where(P.b.str.startswith("orphan"), "orphan", "otherFP"))
print(P.groupby("kind")[cols].median().round(3).to_string())
G["kind"] = np.where(G.tp, "TP", G.b)
print("\nGT medians: TP vs FN buckets")
print(G.groupby("kind")[["area", "elong", "contrast", "flatmin", "r", "alon", "sup"]].median().round(3).to_string())

# GT location: do annotators skip |lon|>70?
print("\n== GT share with |lon|>70:", round(float(np.mean(G.alon > 70)), 3), " pred share:", round(float(np.mean(P.drop_duplicates(['stem','j']).alon > 70)), 3))
for lo, hi in [(0, 30), (30, 50), (50, 60), (60, 70), (70, 80), (80, 91)]:
    p = P[(P.alon >= lo) & (P.alon < hi)]; q = G[(G.alon >= lo) & (G.alon < hi)]
    print(f"  lon {lo}-{hi}: preds {len(p)} precision {p.tp.mean():.2f}  GT {len(q)} recall {q.tp.mean():.2f}")
for lo, hi in [(0, .5), (.5, .8), (.8, .9), (.9, .95), (.95, 1.1)]:
    p = P[(P.r >= lo) & (P.r < hi)]; q = G[(G.r >= lo) & (G.r < hi)]
    print(f"  r {lo}-{hi}: preds {len(p)} precision {p.tp.mean():.2f}  GT {len(q)} recall {q.tp.mean():.2f}")
for lo, hi in [(0, .02), (.02, .05), (.05, .1), (.1, 9)]:
    p = P[(P.sun_d >= lo) & (P.sun_d < hi)]; q = G[(G.sun_d >= lo) & (G.sun_d < hi)]
    print(f"  sun_d {lo}-{hi}: preds {len(p)} precision {p.tp.mean():.2f}  GT {len(q)} recall {q.tp.mean():.2f}")
print("  sun_d nan (no spot):", P.sun_d.isna().mean().round(2))

P.to_pickle(S / "P.pkl"); G.to_pickle(S / "G.pkl")
pickle.dump(dict(S0=S0, TP0=TP0, FP0=FP0, FN0=FN0), open(S / "base.pkl", "wb"))
