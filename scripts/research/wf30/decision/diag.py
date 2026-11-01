import sys, pickle, os
os.environ["SKIP_ORACLE"] = "1"
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
src = open(HERE / "analyze.py").read().split("# ---------- learned selector")[0]
exec(src)
from scipy.stats import spearmanr
P = pickle.load(open(HERE / "oof_r4.pkl", "rb"))["P"]
Ts = TABLES["r4"]
m = np.concatenate([T["v"] / recs[i]["A"] for i, T in enumerate(Ts) if len(T["v"])])
p = np.concatenate([P[i] for i, T in enumerate(Ts) if len(T["v"])])
print(f"nodes {len(m)}  OOF R2 {1-((m-p)**2).sum()/((m-m.mean())**2).sum():.3f} spearman {spearmanr(m,p)[0]:.3f}")
# oracle tree selection with q at its fixed point, inspect chosen nodes
def oracle_sel(q):
    out = []
    for i, T in enumerate(Ts):
        out.append(dp_select(T, T["v"] - q / 2 * recs[i]["A"]) if len(T["v"]) else [])
    return out
q = 0.538
sel = oracle_sel(q)
lv = np.concatenate([Ts[i]["lvl"][c] for i, c in enumerate(sel) if len(c)])
print("oracle chosen levels:", {float(t): int((lv == t).sum()) for t in np.unique(lv)})
# matched IoU of oracle TPs: how many just above 0.5
iouTP = []
for i, c in enumerate(sel):
    for k in c:
        for g in recs[i]["fams"]["r4"][k]["ious"]:
            if len(g) and g.max() > .5: iouTP.append(g.max())
iouTP = np.array(iouTP)
bTP = np.concatenate([m_[m_ > .5] for r in recs for m_ in r["base"]["ious"]])
print(f"oracle TP {len(iouTP)} frac IoU<0.6 {np.mean(iouTP<0.6):.3f} mean {iouTP.mean():.3f}; base TP {len(bTP)} frac<0.6 {np.mean(bTP<0.6):.3f} mean {bTP.mean():.3f}")
# noise test: oracle values + gaussian noise of the OOF residual size -> how fast does the ceiling collapse
res_sd = np.std(m - p)
for sd in (0.02, 0.05, 0.1, res_sd):
    rng = np.random.default_rng(1); best = 0
    for tau in np.arange(0.12, 0.34, 0.02):
        rows = []
        for i, T in enumerate(Ts):
            r = recs[i]
            if not len(T["v"]): rows.append(img_stats(r, 0, 0, 0)); continue
            noisy = T["v"] / r["A"] + rng.normal(0, sd, len(T["v"]))
            c = dp_select(T, noisy - tau)
            rows.append(img_stats(r, T["v"][c].sum(), T["tp"][c].sum(), len(c)))
        best = max(best, pq(np.stack(rows).sum(0)))
    print(f"oracle values + N(0,{sd:.3f}) noise, best tau (in-sample): PQ {best:.4f}")
# per-image global level oracle (single threshold per image, keep all nodes at that level)
for name, keepall in (("keep all", True),):
    tot = np.zeros(4)
    for i, T in enumerate(Ts):
        r = recs[i]; best = None
        for t in LEV if False else np.unique(T["lvl"]) if len(T["lvl"]) else []:
            c = np.nonzero(T["lvl"] == t)[0]
            s = img_stats(r, T["v"][c].sum(), T["tp"][c].sum(), len(c))
            sc = s[0] - q/2*(s[1]+.5*s[2]+.5*s[3])
            if best is None or sc > best[0]: best = (sc, s)
        tot += best[1] if best else img_stats(r, 0, 0, 0)
    print(f"ORACLE per-image single level (keep all at it): PQ {pq(tot):.4f} vs base {pq(BASE.sum(0)):.4f}")
# annotator agreement among base TPs: per instance, number of annotators matched (images with A>=2)
cnt = {}
for r in recs:
    if r["A"] < 2: continue
    h = np.zeros(r["base"]["feats"].shape[0], int)
    for m_ in r["base"]["ious"]:
        if m_.size: h += (m_ > .5).any(0)
    for k in range(r["A"] + 1): cnt[(r["A"], k)] = cnt.get((r["A"], k), 0) + int((h == k).sum())
print("base instances by (A, #annotators matched):", cnt)
