"""Compare pipelines on the same held-out images with a paired bootstrap over images.

Each spec is  name=probs_dir,post.json[,filter.pkl]  or  name=probs_dir,tune:<mode>.
A tune:<mode> spec is cross-tuned: the images are split in two halves, the post-processing grid of
that mode (GRIDS below) is tuned on one half and scored on the other, both ways, over 3 random
splits -- an honest estimate for a pipeline whose parameters are tuned on the same val images.
Without a filter every instance is kept;
with one, the filter is refitted in 5 folds by image (its saved model was fitted on these very
images, so using it directly would be in-sample) and its saved threshold is applied to the
out-of-fold scores. The first spec is the baseline; for every other spec the report gives
PQ, the difference to the baseline with a 95% bootstrap interval, and P(difference > 0).

Usage:
  python scripts/compare.py \
      base=outputs/colab/unet1024_effb4/probs_val,configs/post_unet1024_effb4.json \
      filt=outputs/colab/unet1024_effb4/probs_val,configs/post_unet1024_effb4.json,configs/fpfilter_unet1024_effb4.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from filseg.data import load_train, with_maps  # noqa: E402
from filseg.dl import PostParams, load_maps  # noqa: E402
from filseg.instfilter import select  # noqa: E402
from filseg.metric import IOU_THRESHOLD  # noqa: E402
from fp_filter import _image_table, make_model, predict  # noqa: E402


def image_stats(ious: list[np.ndarray], keep: np.ndarray) -> np.ndarray:
    """(sum TP IoU, TP, FP, FN) of one image, summed over its annotator entries."""
    out = np.zeros(4)
    for m in ious:
        m = m[:, keep]
        hit = m > IOU_THRESHOLD
        out += [m[hit].sum(), hit.sum(), (hit.sum(0) == 0).sum(), (hit.sum(1) == 0).sum()]
    return out


def pq(s: np.ndarray) -> float:
    den = s[1] + 0.5 * s[2] + 0.5 * s[3]
    return s[0] / den if den else 0.0


GRIDS = {
    "cc": {"thresh": [0.2, 0.3, 0.4, 0.5, 0.6], "min_area": [50, 90, 150], "group_radius": [4, 6, 8]},
    "hyst": {"thresh": [0.3, 0.4, 0.5], "seed_thresh": [0.9, 0.95, 0.97, 0.99], "min_area": [50, 90],
             "group_radius": [6]},
    # strict hysteresis + shrinking small instances
    "hyst_size": {"thresh": [0.4, 0.5], "seed_thresh": [0.97, 0.99], "min_area": [50, 90], "group_radius": [4],
                  "small_area": [200], "small_thresh": [0.0, 0.7, 0.8]},
    # cc + size-dependent extent; areas for 1024 px maps (800 / 3200 px at 2048 scale)
    "cc_size": {"thresh": [0.5, 0.6], "min_area": [90], "group_radius": [4, 6],
                "small_area": [200], "small_thresh": [0.0, 0.7, 0.8],
                "large_area": [800], "large_thresh": [0.0, 0.4, 0.5]},
}


# sizes in GRIDS are for 1024 px maps; areas scale with the square of the map resolution
AREA_KEYS, RADIUS_KEYS = ("min_area", "small_area", "large_area"), ("group_radius",)


def scale_grid(grid: dict, res: int) -> dict:
    """GRIDS[mode] with areas and radii rescaled from 1024 px maps to `res` px maps."""
    r = res / 1024
    out = {}
    for k, v in grid.items():
        if k in AREA_KEYS:
            out[k] = [round(a * r * r) for a in v]
        elif k in RADIUS_KEYS:
            out[k] = [round(a * r) for a in v]
        else:
            out[k] = list(v)
    return out


def _combo_stats(args):
    probs, post, stems, jobs_by_stem = args
    jobs = [(probs, post, jobs_by_stem[s][0], s, jobs_by_stem[s][1]) for s in stems]
    rows = []
    for j in jobs:
        t = _image_table(j)
        rows.append(image_stats(t[3], np.ones(len(t[2]), bool)))
    return np.stack(rows)


def evaluate_crosstuned(probs: str, mode: str, stems: list[str], jobs_by_stem: dict,
                        repeats: int = 3) -> np.ndarray:
    """Per-image stats when the grid is tuned on the other half of the images (mean over repeats)."""
    import itertools

    grid = scale_grid(GRIDS[mode], load_maps(Path(probs), stems[0])[0].shape[0])
    base_mode = {"cc_size": "cc", "hyst_size": "hyst"}.get(mode, mode)
    combos = [PostParams(mode=base_mode, **dict(zip(grid, v)))
              for v in itertools.product(*grid.values())]
    with ProcessPoolExecutor() as pool:
        per_combo = np.stack(list(pool.map(_combo_stats, [(probs, c, stems, jobs_by_stem) for c in combos])))
    out = np.zeros((len(stems), 4))
    chosen = []
    for r in range(repeats):
        half = np.random.default_rng(100 + r).permutation(len(stems)) < len(stems) // 2
        for tune_on in (half, ~half):
            best = int(np.argmax([pq(per_combo[c][tune_on].sum(0)) for c in range(len(combos))]))
            chosen.append(best)
            out[~tune_on] += per_combo[best][~tune_on]
    print(f"  {probs} {mode}: combos chosen {[combos[c].to_dict() for c in sorted(set(chosen))][:3]}",
          file=sys.stderr, flush=True)
    return out / repeats


def evaluate(spec: str, stems: list[str], jobs_by_stem: dict, folds: int = 5) -> np.ndarray:
    """Per-image stats (n_images, 4) of one spec, in the order of `stems`."""
    name, rest = spec.split("=", 1)
    parts = rest.split(",")
    if parts[1].startswith("tune:"):
        return evaluate_crosstuned(parts[0], parts[1][5:], stems, jobs_by_stem)
    probs, post = parts[0], PostParams(**json.loads(Path(parts[1]).read_text()))
    filt = pickle.loads(Path(parts[2]).read_bytes()) if len(parts) > 2 else None
    jobs = [(probs, post, jobs_by_stem[s][0], s, jobs_by_stem[s][1]) for s in stems]
    with ProcessPoolExecutor() as pool:
        tables = {t[0]: t for t in pool.map(_image_table, jobs, chunksize=4)}
    keep = {s: np.ones(len(tables[s][2]), bool) for s in stems}
    if filt:
        kind, names = filt.get("kind", "clf"), filt["features"]
        idx = list(range(len(stems)))
        random.Random(0).shuffle(idx)
        fold_of = {stems[i]: f for f, chunk in enumerate(np.array_split(idx, folds)) for i in chunk}
        for f in range(folds):
            tr = [s for s in stems if fold_of[s] != f and len(tables[s][2])]
            te = [s for s in stems if fold_of[s] == f and len(tables[s][2])]
            X = select(np.concatenate([tables[s][1] for s in tr]), names)
            y = np.concatenate([tables[s][2] if kind == "clf" else tables[s][4] for s in tr])
            model = make_model(kind).fit(X, y)
            for s in te:
                keep[s] = predict(model, kind, select(tables[s][1], names)) >= filt["threshold"]
    return np.stack([image_stats(tables[s][3], keep[s]) for s in stems])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    first_probs = args.specs[0].split("=", 1)[1].split(",")[0]
    jobs_by_stem: dict[str, tuple[Path, list]] = {}
    for it in with_maps(load_train(), Path(first_probs)):
        jobs_by_stem.setdefault(it.image_id, (it.path, []))[1].append(it.gt_rles)
    stems = sorted(jobs_by_stem)
    stats = {}
    for spec in args.specs:
        stats[spec.split("=", 1)[0]] = evaluate(spec, stems, jobs_by_stem)
        print(f"evaluated {spec.split('=', 1)[0]}", file=sys.stderr, flush=True)

    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(stems), size=(args.boot, len(stems)))
    names = list(stats)
    base = stats[names[0]]
    base_boot = np.array([pq(base[b].sum(0)) for b in boot])
    print(f"{len(stems)} images, {args.boot} bootstrap resamples")
    print(f"{'pipeline':28s} {'PQ':>7s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'diff':>8s} {'95% CI':>18s} {'P>0':>5s}")
    for n in names:
        s = stats[n]
        tot = s.sum(0)
        line = f"{n:28s} {pq(tot):7.4f} {tot[1]:5.0f} {tot[2]:5.0f} {tot[3]:5.0f}"
        if n != names[0]:
            d = np.array([pq(s[b].sum(0)) for b in boot]) - base_boot
            line += f" {pq(tot) - pq(base.sum(0)):+8.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] {np.mean(d > 0):5.2f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
