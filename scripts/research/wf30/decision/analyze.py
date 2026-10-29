"""Oracle ceilings and image-level cross-validated learned selectors on the candidate trees."""
import sys, pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
HERE = Path(__file__).parent
recs = [pickle.loads(p.read_bytes()) for p in sorted((HERE / "cands").glob("*.pkl"))]
N = len(recs); print("images", N)
G_TOTAL = sum(sum(r["n_gt"]) for r in recs)

def matched(ious_list):
    """per-annotator matched IoU arrays -> (v=sum over annotators, tp count, per-ann hits)"""
    v, tp = 0.0, 0
    for g in ious_list:
        if len(g):
            mx = g.max()
            if mx > 0.5: v += mx; tp += 1
    return v, tp

# per-image node tables
def node_table(r, fam):
    nodes = r["fams"][fam]
    key = {(n["level"], n["k"]): i for i, n in enumerate(nodes)}
    par = np.array([key.get((n["level"] - 1, n["parent_k"]), -1) for n in nodes])
    ch = [[] for _ in nodes]
    for i, p in enumerate(par):
        if p >= 0: ch[p].append(i)
    v = np.zeros(len(nodes)); tp = np.zeros(len(nodes))
    for i, n in enumerate(nodes):
        v[i], tp[i] = matched(n["ious"])
    area = np.array([n["area"] for n in nodes], float)
    lvl = np.array([n["t"] for n in nodes])
    pa = np.where(par >= 0, area[np.maximum(par, 0)], np.nan)
    nch = np.array([len(c) for c in ch])
    chfrac = np.array([area[c].sum() / area[i] if c else 0 for i, c in enumerate(ch)])
    # persistence: how many consecutive levels this lineage keeps a single child
    F = np.array([n["feats"] for n in nodes]) if nodes else np.zeros((0, 17))
    extra = np.c_[lvl, area / np.where(np.isnan(pa), area, pa), nch, chfrac,
                  np.where(par >= 0, 1, 0)]
    X = np.c_[F, extra] if len(nodes) else np.zeros((0, 22))
    return dict(par=par, ch=ch, v=v, tp=tp, X=X, lvl=lvl, area=area)

def dp_select(T, w):
    """max-weight antichain in the forest; returns chosen node idxs."""
    n = len(w); best = np.zeros(n); take = np.zeros(n, bool)
    order = np.argsort(-T["lvl"], kind="stable")  # children (higher level) first
    for i in order:
        s = sum(best[c] for c in T["ch"][i])
        if w[i] > s and w[i] > 0: best[i] = w[i]; take[i] = True
        else: best[i] = max(s, 0.0)
    chosen = []
    stack = [i for i in range(n) if T["par"][i] < 0]
    while stack:
        i = stack.pop()
        if take[i]: chosen.append(i)
        else: stack.extend(T["ch"][i])
    return chosen

def img_stats(r, v_sel, tp_sel, nsel):
    A = r["A"]; G = sum(r["n_gt"])
    S = v_sel; TP = tp_sel; FP = A * nsel - TP; FN = G - TP
    return np.array([S, TP, FP, FN])

def pq(s):
    return s[0] / (s[1] + .5 * s[2] + .5 * s[3])

def base_stats(r, keep=None):
    out = np.zeros(4)
    for m in r["base"]["ious"]:
        if keep is not None: m = m[:, keep]
        hit = m > .5
        out += [m[hit].sum(), hit.sum(), (hit.sum(0) == 0).sum(), (hit.sum(1) == 0).sum()]
    return out

BASE = np.stack([base_stats(r) for r in recs])
print(f"baseline hyst (fixed config, in-sample tuned): PQ {pq(BASE.sum(0)):.4f} TP {BASE.sum(0)[1]:.0f} FP {BASE.sum(0)[2]:.0f} FN {BASE.sum(0)[3]:.0f}")
rng = np.random.default_rng(0); BOOT = rng.integers(0, N, (2000, N))
def report(name, ST):
    tot = ST.sum(0); d = np.array([pq(ST[b].sum(0)) - pq(BASE[b].sum(0)) for b in BOOT])
    print(f"{name:52s} PQ {pq(tot):.4f} TP {tot[1]:5.0f} FP {tot[2]:5.0f} FN {tot[3]:5.0f} d {pq(tot)-pq(BASE.sum(0)):+.4f} [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]", flush=True)
    return ST

# base instance values
for r in recs:
    ious = r["base"]["ious"]; n = r["base"]["feats"].shape[0]
    v = np.zeros(n); tp = np.zeros(n)
    for m in ious:
        if m.size:
            hit = m > .5; v += np.where(hit, m, 0).max(0); tp += hit.any(0)
    r["bv"], r["btp"] = v, tp

def dinkel(per_img_fn):
    q = pq(BASE.sum(0))
    for _ in range(20):
        ST = np.stack([per_img_fn(i, q) for i in range(N)])
        nq = pq(ST.sum(0))
        if abs(nq - q) < 1e-7: break
        q = nq
    return ST

# Oracle 1: keep/drop on baseline instances
def o_keep(i, q):
    r = recs[i]; keep = r["bv"] - q / 2 * r["A"] > 0
    return base_stats(r, keep)
import os
SKIP=os.environ.get("SKIP_ORACLE")
if not SKIP: report("ORACLE keep/drop on hyst instances", dinkel(o_keep))

TABLES = {fam: [node_table(r, fam) for r in recs] for fam in ("r0", "r4")}
for fam in ("r0", "r4"):
    Ts = TABLES[fam]
    def o_tree(i, q, Ts=Ts):
        r, T = recs[i], Ts[i]
        if not len(T["v"]): return img_stats(r, 0, 0, 0)
        c = dp_select(T, T["v"] - q / 2 * r["A"])
        return img_stats(r, T["v"][c].sum(), T["tp"][c].sum(), len(c))
    if not SKIP: report(f"ORACLE tree antichain fam={fam} (levels+split+keep)", dinkel(o_tree))
    for t in (0.5,):
        def o_lvl(i, q, Ts=Ts, t=t):
            r, T = recs[i], Ts[i]
            sel = np.nonzero(np.isclose(T["lvl"], t))[0]
            w = T["v"][sel] - q / 2 * r["A"]; c = sel[w > 0]
            return img_stats(r, T["v"][c].sum(), T["tp"][c].sum(), len(c))
        if not SKIP: report(f"ORACLE keep/drop single level {t} fam={fam}", dinkel(o_lvl))
    # oracle extent only: keep ALL roots-equivalents? -> 'every chosen lineage': force choose one per root that baseline-level 0.5 has
# number of annotators distribution
A = np.array([r["A"] for r in recs]); print("annotators per image:", np.bincount(A))

# ---------- learned selector (5-fold by image OOF) ----------
def oof_predict(fam, target="m", seed=0):
    Ts = TABLES[fam]
    folds = np.random.default_rng(seed).permutation(N) % 5
    preds = [None] * N
    for f in range(5):
        tr = [i for i in range(N) if folds[i] != f and len(Ts[i]["v"])]
        X = np.concatenate([Ts[i]["X"] for i in tr])
        y = np.concatenate([Ts[i]["v"] / recs[i]["A"] for i in tr])
        mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.07, max_leaf_nodes=15,
                                            l2_regularization=1.0, random_state=seed).fit(X, y)
        for i in range(N):
            if folds[i] == f and len(Ts[i]["v"]): preds[i] = mdl.predict(Ts[i]["X"])
    return preds

def crosstune(stats_by_tau):
    """stats_by_tau: (n_tau, N, 4). Tune tau on half, score other half, 3 repeats."""
    out = np.zeros((N, 4)); taus = []
    for rep in range(3):
        half = np.random.default_rng(100 + rep).permutation(N) < N // 2
        for on in (half, ~half):
            b = int(np.argmax([pq(s[on].sum(0)) for s in stats_by_tau])); taus.append(b)
            out[~on] += stats_by_tau[b][~on]
    return out / 3, taus

TAUS = np.arange(0.10, 0.34, 0.03)
for fam in ("r0", "r4"):
    Ts = TABLES[fam]; P = oof_predict(fam)
    sbt = []
    for tau in TAUS:
        rows = []
        for i in range(N):
            r, T = recs[i], Ts[i]
            if not len(T["v"]): rows.append(img_stats(r, 0, 0, 0)); continue
            c = dp_select(T, P[i] - tau)
            rows.append(img_stats(r, T["v"][c].sum(), T["tp"][c].sum(), len(c)))
        sbt.append(np.stack(rows))
    sbt = np.stack(sbt)
    print(fam, "in-sample tau curve:", " ".join(f"{t:.2f}:{pq(s.sum(0)):.4f}" for t, s in zip(TAUS, sbt)))
    ST, taus = crosstune(sbt)
    report(f"LEARNED tree selector fam={fam} (OOF GBM, tau cross-tuned)", ST)
    pickle.dump(dict(P=P), open(HERE / f"oof_{fam}.pkl", "wb"))

# learned keep/drop on base instances (same machinery) for comparison
folds = np.random.default_rng(0).permutation(N) % 5
bp = [None] * N
for f in range(5):
    tr = [i for i in range(N) if folds[i] != f and len(recs[i]["bv"])]
    X = np.concatenate([recs[i]["base"]["feats"] for i in tr]); y = np.concatenate([recs[i]["bv"] / recs[i]["A"] for i in tr])
    mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.07, max_leaf_nodes=15, l2_regularization=1.0, random_state=0).fit(X, y)
    for i in range(N):
        if folds[i] == f and len(recs[i]["bv"]): bp[i] = mdl.predict(recs[i]["base"]["feats"])
sbt = np.stack([np.stack([base_stats(r, (bp[i] > tau) if bp[i] is not None else None) for i, r in enumerate(recs)]) for tau in TAUS])
ST, _ = crosstune(sbt); report("LEARNED keep/drop on hyst instances (OOF, tau xtuned)", ST)
