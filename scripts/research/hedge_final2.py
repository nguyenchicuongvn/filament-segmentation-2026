"""Operating point of the final pool under annotator shift (hedge for final 2).

The public LB sits ~0.05 under OOF and reacted to dropping instances (-0.02 for 15%) more than OOF
predicted, so the test set may reward recall more than the pooled OOF does (RESEARCH H5: other
annotator mix). For a fixed list of post-processing variants (declared before looking at numbers)
and FP-filter thresholds (filter refitted per variant, 5-fold CV by image as scripts/fp_filter.py),
print PQ under several views of the 707 OOF images:
  all      every annotator entry (the selection metric)
  one      one random annotator per image (mean of 50 draws)
  single   images with one annotator / multi: images with 2-3
  g01..06  entries of one annotator group (first two digits of the annotator id)
  hard     groups 01 + 04 (lowest PQ, H5)
and the paired bootstrap of each (variant, threshold) against the base under `all`.

  python scripts/research/hedge_final2.py --pool outputs/ens/abl/n1n9p_oof --base-th 0.275
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filseg.data import load_train, with_maps  # noqa: E402
from filseg.dl import PostParams  # noqa: E402
from filseg.metric import IOU_THRESHOLD  # noqa: E402

_spec = importlib.util.spec_from_file_location("fp_filter", ROOT / "scripts" / "fp_filter.py")
fpf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fpf)

BASE = PostParams(thresh=0.5, min_area=200, group_radius=6, group_line=16, mode="hyst", seed_thresh=0.99)
VARIANTS = {
    "base": {},
    "s95": dict(seed_thresh=0.95),
    "s90": dict(seed_thresh=0.90),
    "a100": dict(min_area=100),
    "s95a100": dict(seed_thresh=0.95, min_area=100),
    "t40": dict(thresh=0.4),
    "t40s95a100": dict(thresh=0.4, seed_thresh=0.95, min_area=100),
    "s80a50": dict(seed_thresh=0.80, min_area=50),
}
THS = [-1.0, 0.1, 0.2, 0.275, 0.375]


def _table(job):
    return fpf._image_table(job)


def pq(s):
    s = np.asarray(s).reshape(-1, 4).sum(0)
    den = s[1] + 0.5 * s[2] + 0.5 * s[3]
    return s[0] / den if den else 0.0


def entry_stats(tables, oof, owner, th):
    """One (sum TP IoU, TP, FP, FN) row per annotator entry, in table order."""
    rows = []
    for i, t in enumerate(tables):
        keep = oof[owner == i] >= th
        for m in t[3]:
            m = m[:, keep]
            hit = m > IOU_THRESHOLD
            rows.append([m[hit].sum(), hit.sum(), (hit.sum(0) == 0).sum(), (hit.sum(1) == 0).sum()])
    return np.asarray(rows, float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--base-th", type=float, default=0.275, help="filter threshold of the current final")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--cache", type=Path, default=ROOT / "outputs" / "cache_hedge")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    by_img: dict[str, tuple[Path, list, list]] = {}
    for it in with_maps(load_train(), Path(args.pool)):
        e = by_img.setdefault(it.image_id, (it.path, [], []))
        e[1].append(it.gt_rles)
        e[2].append(it.key)
    stems = sorted(by_img)
    keys = [k for s in stems for k in by_img[s][2]]
    img_of = np.asarray([i for i, s in enumerate(stems) for _ in by_img[s][2]])
    group = np.asarray([k[:2] for k in keys])
    n_ann = np.bincount(img_of)[img_of]
    print(f"{len(stems)} images, {len(keys)} entries; groups "
          + " ".join(f"{g}:{(group == g).sum()}" for g in sorted(set(group))), flush=True)

    idx = list(range(len(stems)))
    random.Random(0).shuffle(idx)
    fold_of = np.empty(len(stems), int)
    for f, chunk in enumerate(np.array_split(idx, 5)):
        fold_of[chunk] = f

    rng = np.random.default_rng(0)
    one_draws = []
    starts = np.r_[0, np.cumsum(np.bincount(img_of))[:-1]]
    counts = np.bincount(img_of)
    for _ in range(50):
        one_draws.append(starts + rng.integers(0, counts))
    views = {"all": np.ones(len(keys), bool), "single": n_ann == 1, "multi": n_ann > 1,
             "hard": np.isin(group, ["01", "04"])}
    views.update({f"g{g}": group == g for g in sorted(set(group))})

    stats = {}
    for name, kw in VARIANTS.items():
        post = replace(BASE, **kw)
        cf = args.cache / f"{Path(args.pool).name}_{name}.pkl"
        if cf.exists():
            tables = pickle.loads(cf.read_bytes())
        else:
            jobs = [(args.pool, post, by_img[s][0], s, by_img[s][1]) for s in stems]
            with ProcessPoolExecutor(args.jobs) as ex:
                tables = list(ex.map(_table, jobs, chunksize=4))
            cf.write_bytes(pickle.dumps(tables))
        X = np.concatenate([t[1] for t in tables])
        y = np.concatenate([t[2] for t in tables])
        owner = np.repeat(np.arange(len(tables)), [len(t[2]) for t in tables])
        oof = np.zeros(len(y))
        for f in range(5):
            te = fold_of[owner] == f
            oof[te] = fpf.make_model("clf").fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
        for th in THS:
            stats[name, th] = entry_stats(tables, oof, owner, th)
        print(f"{name}: {len(y)} instances", flush=True)

    head = ["all", "one", "single", "multi", "hard"] + [v for v in views if v.startswith("g")]
    print(f"\n{'variant':12s} {'th':>6s} " + " ".join(f"{h:>7s}" for h in head) + "   TP    FP    FN")
    for (name, th), s in stats.items():
        vals = [pq(s[views["all"]]), float(np.mean([pq(s[d]) for d in one_draws]))]
        vals += [pq(s[views[h]]) for h in head[2:]]
        c = s.sum(0)[1:].astype(int)
        print(f"{name:12s} {th:6.3f} " + " ".join(f"{v:7.4f}" for v in vals) + f"  {c[0]:5d} {c[1]:5d} {c[2]:5d}")

    # paired bootstrap over images against the current final (base, base-th), view `all` and `hard`
    ref = stats["base", args.base_th]
    n = len(stems)

    def img_sum(s, mask):
        out = np.zeros((n, 4))
        np.add.at(out, img_of[mask], s[mask])
        return out
    print(f"\npaired vs base@{args.base_th} (bootstrap {args.boot} over images): all | hard")
    draws = rng.integers(0, n, (args.boot, n))
    for (name, th), s in stats.items():
        line = f"{name:12s} {th:6.3f}"
        for v in ("all", "hard"):
            a, b = img_sum(ref, views[v]), img_sum(s, views[v])
            ds = np.array([pq(b[i]) - pq(a[i]) for i in draws])
            lo, hi = np.percentile(ds, [2.5, 97.5])
            line += f"  {pq(b) - pq(a):+.4f} [{lo:+.4f}, {hi:+.4f}]"
        print(line)


if __name__ == "__main__":
    main()
