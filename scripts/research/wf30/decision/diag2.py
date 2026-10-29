import os, pickle
os.environ["SKIP_ORACLE"] = "1"
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
exec(open(HERE / "analyze.py").read().split("# ---------- learned selector")[0])
TAUS = np.arange(0.10, 0.34, 0.03)
def crosstune(sbt):
    out=np.zeros((N,4))
    for rep in range(3):
        half=np.random.default_rng(100+rep).permutation(N)<N//2
        for on in (half,~half):
            b=int(np.argmax([pq(s[on].sum(0)) for s in sbt])); out[~on]+=sbt[b][~on]
    return out/3, None
P = pickle.load(open(HERE / "oof_r4.pkl", "rb"))["P"]
def subset(T, levels):
    keep = np.isin(np.round(T["lvl"], 2), levels); idx = np.nonzero(keep)[0]; pos = {j: n for n, j in enumerate(idx)}
    par = []
    for j in idx:
        a = T["par"][j]
        while a >= 0 and not keep[a]: a = T["par"][a]
        par.append(pos[a] if a >= 0 else -1)
    par = np.array(par, int); ch = [[] for _ in idx]
    for n, a in enumerate(par):
        if a >= 0: ch[a].append(n)
    return dict(par=par, ch=ch, v=T["v"][idx], tp=T["tp"][idx], lvl=T["lvl"][idx]), idx
for levels in ([0.5], [0.3, 0.5, 0.7], [0.3, 0.5, 0.7, 0.9], [0.2, 0.5, 0.8, 0.95]):
    subs = [subset(T, levels) for T in TABLES["r4"]]
    def o(i, q):
        T = subs[i][0]; r = recs[i]
        if not len(T["v"]): return img_stats(r, 0, 0, 0)
        c = dp_select(T, T["v"] - q / 2 * r["A"]); return img_stats(r, T["v"][c].sum(), T["tp"][c].sum(), len(c))
    report(f"ORACLE tree r4 levels {levels}", dinkel(o))
    sbt = []
    for tau in TAUS:
        rows = []
        for i in range(N):
            T, idx = subs[i]; r = recs[i]
            if not len(T["v"]): rows.append(img_stats(r, 0, 0, 0)); continue
            c = dp_select(T, P[i][idx] - tau); rows.append(img_stats(r, T["v"][c].sum(), T["tp"][c].sum(), len(c)))
        sbt.append(np.stack(rows))
    ST, _ = crosstune(np.stack(sbt)); report(f"LEARNED tree r4 levels {levels}", ST)
