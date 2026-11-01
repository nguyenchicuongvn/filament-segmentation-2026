"""Lighter learned selection: 5-fold OOF predictions (one GBM fit per fold), PQ reported for every
lambda in a small grid; lambda chosen by cross-tuning (half of images picks, other half scores)."""
import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.argv = [sys.argv[0], sys.argv[1]]
exec(open(Path(__file__).parent / "tree_select.py").read().split("# learned, 5-fold CV by image")[0].replace("# oracle (Dinkelbach)", "if False:\n  pass\n# oracle skipped").replace("for it in range(6):", "for it in range(0):").replace('print("ORACLE', '#').replace("for lv in (0.4, 0.5, 0.6):", "for lv in ():"))
rng = np.random.default_rng(42)
perm = list(rng.permutation(stems))
folds = [set(perm[i::5]) for i in range(5)]
oof = pd.Series(0.0, index=D.index)
for f in folds:
    tr = D[~D.stem.isin(f)]; te = D[D.stem.isin(f)]
    mdl = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
                                        min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    mdl.fit(tr[FEAT], tr.y); oof[te.index] = mdl.predict(te[FEAT]); print("fold done", flush=True)
D["oof"] = oof
sc = {s: g.sort_values("idx").oof.values for s, g in D.groupby("stem")}
res = {}
for lm in [0.35, 0.45, 0.55, 0.65]:
    sel = {s: (select(by[s], sc[s] - lm * 0.5) if s in sc else []) for s in by}
    res[lm] = evaluate(sel)
    d, lo, hi = boot(base_img, res[lm])
    print(f"lambda {lm:.2f}: PQ {pq(res[lm]):.4f}  vs base {d:+.4f} [{lo:+.4f},{hi:+.4f}]  n_sel {sum(len(v) for v in sel.values())}")
# cross-tune lambda: pick on half, score other half, 3 splits x 2 directions
num = den = 0.0; ks = list(by)
out = {}
for sd in range(3):
    r2 = np.random.default_rng(sd); p2 = list(r2.permutation(ks)); A, B = p2[::2], p2[1::2]
    for a, b in ((A, B), (B, A)):
        lm = max(res, key=lambda l: pq(res[l], a))
        for k in b:
            out.setdefault(k, []).append(res[lm][k])
ct = {k: tuple(np.mean(v, axis=0)) for k, v in out.items()}
d, lo, hi = boot(base_img, ct)
print(f"CROSS-TUNED lambda: PQ {pq(ct):.4f} vs base {d:+.4f} [{lo:+.4f},{hi:+.4f}]")
pickle.dump(dict(res=res, ct=ct, oof=D[["stem", "idx", "oof", "y"]]), open(S / "learned_fast.pkl", "wb"))
