"""PQ of TTA variants (fixed hyst config tuned on base maps -> biased toward base) on the images
where all variants exist. Paired bootstrap by image vs base. Also blur effect by hp tercile."""
import json, pickle, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from filseg.data import by_image, encode_masks, load_train, with_maps  # noqa
from filseg.dl import PostParams, instances_lowres, upsample_labels  # noqa
from filseg.metric import PQAccumulator  # noqa

T = Path(__file__).parent / "tta"
POST = PostParams(**json.loads((ROOT / "configs/post_unet1024_effb4_hyst.json").read_text()))
VARS = {"base": ["base"], "blur": ["blur"], "s1152": ["s1152"], "s896": ["s896"],
        "base+s1152": ["base", "s1152"], "base+s896+s1152": ["base", "s896", "s1152"],
        "base+blur": ["base", "blur"]}


def job(a):
    stem, gts = a
    maps = {v: np.load(T / v / f"{stem}.npy").astype(np.float32) / 255 for v in ["base", "blur", "s1152", "s896"]}
    out = {}
    for name, vs in VARS.items():
        p = np.mean([maps[v] for v in vs], 0)
        pred = encode_masks(upsample_labels(instances_lowres(p, POST)))
        acc = PQAccumulator()
        for g in gts:
            acc.add(g, pred)
        out[name] = (sum(acc.tp_iou), acc.tp, acc.fp, acc.fn)
    return stem, out


if __name__ == "__main__":
    have = set.intersection(*[{f.stem for f in (T / v).glob("*.npy")} for v in ["base", "blur", "s1152", "s896"]])
    g = by_image(load_train())
    jobs = [(s, [t.gt_rles for t in g[s]]) for s in sorted(have)]
    with ProcessPoolExecutor(3) as ex:
        res = dict(ex.map(job, jobs))
    stems = sorted(res)
    arr = {n: np.array([res[s][n] for s in stems]) for n in VARS}
    pq = lambda a: a[:, 0].sum() / (a[:, 1].sum() + .5 * a[:, 2].sum() + .5 * a[:, 3].sum())
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(stems), len(stems)) for _ in range(2000)]
    print(f"{len(stems)} images (fixed post config tuned on base maps)")
    for n, a in arr.items():
        d = [pq(a[i]) - pq(arr["base"][i]) for i in idx]
        print(f"  {n:16s} PQ {pq(a):.4f}  TP {a[:,1].sum()} FP {a[:,2].sum()} FN {a[:,3].sum()}  d {pq(a)-pq(arr['base']):+.4f} CI [{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]")
    import pandas as pd
    st = pd.read_csv(Path(__file__).parent / "station_stats.csv").set_index("stem")
    hp = np.array([st.loc[s, "hp"] for s in stems])
    q = np.quantile(st[st.split == "train"].hp, [1 / 3, 2 / 3])
    t = np.digitize(hp, q)
    for k in range(3):
        m = t == k
        print(f"  hp tercile {k} (n={m.sum()}): base {pq(arr['base'][m]):.4f} blur {pq(arr['blur'][m]):.4f} base+blur {pq(arr['base+blur'][m]):.4f} base+s1152 {pq(arr['base+s1152'][m]):.4f}")
    pickle.dump(res, open(Path(__file__).parent / "eval_tta.pkl", "wb"))
