"""Human-vs-human PQ (every ordered annotator pair of every multi-annotator train image, as F6)
split by image high-frequency energy tercile (hp from station_stats.csv), and per station.
If the PQ drop on 'sharp/noisy' images is the same for humans, it is label ambiguity, not a model
input problem."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from filseg.data import by_image, load_train  # noqa
from filseg.metric import PQAccumulator  # noqa

st = pd.read_csv(Path(__file__).parent / "station_stats.csv").set_index("stem")
tr = st[st.split == "train"]
q = {c: np.quantile(tr[c], [1 / 3, 2 / 3]) for c in ["hp", "lapv"]}
rows = []
for s, es in by_image(load_train()).items():
    if len(es) < 2:
        continue
    for a in es:
        for b in es:
            if a is b:
                continue
            acc = PQAccumulator()
            acc.add(a.gt_rles, b.gt_rles)
            rows.append(dict(stem=s, st=s[-2], siou=sum(acc.tp_iou), tp=acc.tp, fp=acc.fp, fn=acc.fn,
                             hp=np.digitize(st.loc[s, "hp"], q["hp"]), lapv=np.digitize(st.loc[s, "lapv"], q["lapv"])))
d = pd.DataFrame(rows)
pq = lambda x: x.siou.sum() / (x.tp.sum() + .5 * x.fp.sum() + .5 * x.fn.sum())
print(f"human-vs-human PQ all multi-ann images: {pq(d):.4f} ({d.stem.nunique()} images)")
for c in ["hp", "lapv"]:
    print(c, {k: f"{pq(x):.3f} (n_img {x.stem.nunique()})" for k, x in d.groupby(c)})
print("station", {k: f"{pq(x):.3f}" for k, x in d.groupby("st")})
import pickle
e = pickle.load(open(Path(__file__).parent / "tables.pkl", "rb"))["ent"]
for c in ["hp", "lapv"]:
    e[c] = [np.digitize(st.loc[s, c], q[c]) for s in e.stem]
    print("model E9a holdout", c, {k: f"{pq(x):.3f} (n_img {x.stem.nunique()})" for k, x in e.groupby(c)})
    m = e[e.nann > 1]
    print("  model, multi-ann images only", c, {k: f"{pq(x):.3f}" for k, x in m.groupby(c)})
