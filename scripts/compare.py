"""Compare pipelines on the same held-out images with a paired bootstrap over images.

Each spec is  name=probs_dir,post.json[,filter.pkl]  or  name=probs_dir,tune:<mode>[,filter:<set>].
A tune:<mode> spec is cross-tuned: the images are split in two halves, the post-processing grid of
that mode (GRIDS below) is tuned on one half and scored on the other, both ways, over 3 random
splits -- an honest estimate for a pipeline whose parameters are tuned on the same val images.
Without a filter every instance is kept;
with one, the filter is refitted in 5 folds by image (its saved model was fitted on these very
images, so using it directly would be in-sample) and its saved threshold is applied to the
out-of-fold scores. A tune:<mode>,filter:<set> spec also learns the instance filter inside the
cross-tuning: in each split the grid is tuned on half A, a classifier on the features <set>
('base' = FEATURES, 'base+agree' = FEATURES + AGREE_FEATURES, which needs --agree-maps) is fitted
on half A's instances, its drop threshold is chosen on half A (5-fold out-of-fold scores by image,
PQ sweep of fp_filter.py), and half B is scored with all three; nothing is fitted on the images it
scores. The first spec is the baseline; for every other spec the report gives
PQ, the difference to the baseline with a 95% bootstrap interval, and P(difference > 0).

Usage:
  python scripts/compare.py \
      base=outputs/colab/unet1024_effb4/probs_val,configs/post_unet1024_effb4.json \
      filt=outputs/colab/unet1024_effb4/probs_val,configs/post_unet1024_effb4.json,configs/fpfilter_unet1024_effb4.pkl
  python scripts/compare.py base=outputs/colab/unet1024_effb4/probs_val,tune:hyst \
      agree=outputs/colab/unet1024_effb4/probs_val,tune:hyst,filter:base+agree \
      --agree-maps outputs/colab/unet1024_r34/probs_val,outputs/colab/unet1024_r34_grp/probs_val,... --jobs 4

Mask R-CNN member (PLAN card N4), detections cached by scripts/infer_detections.py:
  name=<dets_dir>,tune:det                    the detector alone (DET_GRID cross-tuned)
  name=<probs_dir>,tune:fuse,dets:<dets_dir>  U-Net pixels + detector grouping (filseg.maskrcnn.fuse_with_unet):
      in each split the U-Net hysteresis grid is tuned on half A first (exactly as tune:hyst does), then the
      fusion grid on half A given that choice, and half B is scored with both.
The first spec must be a U-Net probs dir (it defines the images and their annotators).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from filseg.data import encode_masks, load_train, with_maps  # noqa: E402
from filseg.dl import PostParams, instances_lowres, load_maps, upsample_labels  # noqa: E402
from filseg.maskrcnn import DetParams, FuseParams, detections_to_labels, fuse_with_unet, load_detections  # noqa: E402
from filseg.instfilter import AGREE_FEATURES, FEATURES, select  # noqa: E402
from filseg.metric import IOU_THRESHOLD  # noqa: E402
from fp_filter import _agree_table, _image_table, make_model, predict  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402

JOBS: int | None = None        # worker processes (None = all cores)
CACHE: Path | None = None      # pickles of per-combo stats and filter tables (None = no cache)
AGREE_MAPS: list[str] = []     # val maps of the 4 other models, for filter:base+agree
FEATURE_SETS = {"base": FEATURES, "base+agree": FEATURES + AGREE_FEATURES}


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
    # soft-target maps estimate the annotator fraction, so a filament 2 of 3 annotators drew peaks near
    # 0.67 and a seed >= 0.9 drops it although it is worth keeping (RESEARCH F26). Pre-registered
    # 2026-11-03 for every soft-target comparison (G1, G2), both arms alike.
    "hyst_soft": {"thresh": [0.2, 0.25, 0.3, 0.4, 0.5], "seed_thresh": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
                  "min_area": [50, 90], "group_radius": [6]},
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


def _cached(key: tuple, compute):
    """compute(), memoised on disk under CACHE by a hash of `key` (always recomputed without CACHE)."""
    if CACHE is None:
        return compute()
    path = CACHE / f"{hashlib.sha1(repr(key).encode()).hexdigest()[:16]}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    out = compute()
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(out))
    tmp.replace(path)
    return out


_GRID_MEMO: dict = {}


def _grid_stats(probs: str, mode: str, stems: list[str], jobs_by_stem: dict) -> tuple[list, np.ndarray]:
    """The grid of `mode` and per-image stats of every combo, (n_combos, n_images, 4); computed once
    per process (several specs on the same maps share it)."""
    import itertools

    grid = scale_grid(GRIDS[mode], load_maps(Path(probs), stems[0])[0].shape[0])
    base_mode = {"cc_size": "cc", "hyst_size": "hyst", "hyst_soft": "hyst"}.get(mode, mode)
    combos = [PostParams(mode=base_mode, **dict(zip(grid, v)))
              for v in itertools.product(*grid.values())]
    key = ("grid", probs, mode, [c.to_dict() for c in combos], stems)
    if repr(key) not in _GRID_MEMO:
        def compute():
            with ProcessPoolExecutor(JOBS) as pool:
                return np.stack(list(pool.map(_combo_stats, [(probs, c, stems, jobs_by_stem) for c in combos])))
        _GRID_MEMO[repr(key)] = _cached(key, compute)
    return combos, _GRID_MEMO[repr(key)]


def _splits(n: int, repeats: int = 3):
    """The cross-tuning splits: tune_on masks of `repeats` random halves, both directions."""
    for r in range(repeats):
        half = np.random.default_rng(100 + r).permutation(n) < n // 2
        yield half
        yield ~half


def evaluate_crosstuned(probs: str, mode: str, stems: list[str], jobs_by_stem: dict,
                        repeats: int = 3) -> np.ndarray:
    """Per-image stats when the grid is tuned on the other half of the images (mean over repeats)."""
    combos, per_combo = _grid_stats(probs, mode, stems, jobs_by_stem)
    out = np.zeros((len(stems), 4))
    chosen = []
    for tune_on in _splits(len(stems), repeats):
        best = int(np.argmax([pq(per_combo[c][tune_on].sum(0)) for c in range(len(combos))]))
        chosen.append(best)
        out[~tune_on] += per_combo[best][~tune_on]
    print(f"  {probs} {mode}: combos chosen {[combos[c].to_dict() for c in sorted(set(chosen))][:3]}",
          file=sys.stderr, flush=True)
    return out / repeats


# ---------------------------------------------------------------- Mask R-CNN member (N4)

# areas at 1024 px; scaled with the detection resolution like GRIDS
DET_GRID = {"score_thresh": [0.3, 0.5, 0.7, 0.85], "min_visible": [0.3, 0.5, 0.7], "min_area": [50]}
FUSE_GRID = {"score_thresh": [0.5, 0.7, 0.9], "claim": [0.3], "merge": [False, True],
             "add_thresh": [1.01, 0.9], "require_det": [False, True]}


def _labels_stats(labels: np.ndarray, gts: list) -> np.ndarray:
    """image_stats of a low-resolution instance label image against the image's annotator entries."""
    rles = encode_masks(upsample_labels(labels))
    ious = [np.asarray(mask_util.iou(rles, gt, [0] * len(gt))).T if gt and rles else np.zeros((len(gt), len(rles)))
            for gt in gts]
    return image_stats(ious, np.ones(len(rles), bool))


def _resized(masks: np.ndarray, res: int) -> np.ndarray:
    import cv2
    if not len(masks) or masks.shape[1] == res:
        return masks
    return np.stack([cv2.resize(m, (res, res), interpolation=cv2.INTER_LINEAR) for m in masks])


def _det_job(args):
    dets_dir, params, stems, jobs_by_stem = args
    rows = []
    for s in stems:
        scores, masks = load_detections(Path(dets_dir), s)
        rows.append(_labels_stats(detections_to_labels(scores, masks, params), jobs_by_stem[s][1]))
    return np.stack(rows)


def _fuse_job(args):
    probs, post, dets_dir, params, stems, jobs_by_stem = args
    rows = []
    for s in stems:
        prob, group = load_maps(Path(probs), s)
        inst = instances_lowres(prob, post, group)
        scores, masks = load_detections(Path(dets_dir), s)
        rows.append(_labels_stats(fuse_with_unet(inst, scores, _resized(masks, inst.shape[0]), params),
                                  jobs_by_stem[s][1]))
    return np.stack(rows)


def _combos(grid: dict, cls):
    import itertools
    return [cls(**dict(zip(grid, v))) for v in itertools.product(*grid.values())]


def evaluate_crosstuned_det(dets_dir: str, stems: list[str], jobs_by_stem: dict) -> np.ndarray:
    res = load_detections(Path(dets_dir), stems[0])[1].shape[-1]
    grid = scale_grid(DET_GRID, res)
    combos = _combos(grid, DetParams)
    key = ("det", dets_dir, [c.to_dict() for c in combos], stems)
    per_combo = _cached(key, lambda: _pool_stack(_det_job, [(dets_dir, c, stems, jobs_by_stem) for c in combos]))
    out, chosen = np.zeros((len(stems), 4)), []
    for tune_on in _splits(len(stems)):
        best = int(np.argmax([pq(per_combo[c][tune_on].sum(0)) for c in range(len(combos))]))
        chosen.append(best)
        out[~tune_on] += per_combo[best][~tune_on]
    print(f"  {dets_dir} det: combos chosen {[combos[c].to_dict() for c in sorted(set(chosen))][:3]}",
          file=sys.stderr, flush=True)
    return out / 3


def evaluate_crosstuned_fuse(probs: str, dets_dir: str, stems: list[str], jobs_by_stem: dict) -> np.ndarray:
    hyst_combos, hyst_stats = _grid_stats(probs, "hyst", stems, jobs_by_stem)
    res = load_maps(Path(probs), stems[0])[0].shape[0]
    fuse_combos = _combos({**FUSE_GRID, "min_area": [round(50 * (res / 1024) ** 2)]}, FuseParams)
    fuse_stats = {}   # hyst combo index -> (n_fuse, n_images, 4)
    out, chosen = np.zeros((len(stems), 4)), []
    for tune_on in _splits(len(stems)):
        h = int(np.argmax([pq(hyst_stats[c][tune_on].sum(0)) for c in range(len(hyst_combos))]))
        if h not in fuse_stats:
            post = hyst_combos[h]
            key = ("fuse", probs, dets_dir, post.to_dict(), [c.to_dict() for c in fuse_combos], stems)
            fuse_stats[h] = _cached(key, lambda: _pool_stack(
                _fuse_job, [(probs, post, dets_dir, c, stems, jobs_by_stem) for c in fuse_combos]))
        f = int(np.argmax([pq(fuse_stats[h][c][tune_on].sum(0)) for c in range(len(fuse_combos))]))
        chosen.append((h, f))
        out[~tune_on] += fuse_stats[h][f][~tune_on]
    print(f"  {probs} + {dets_dir} fuse: chosen {[(hyst_combos[h].to_dict(), fuse_combos[f].to_dict()) for h, f in sorted(set(chosen))][:2]}",
          file=sys.stderr, flush=True)
    return out / 3


def _pool_stack(fn, jobs: list) -> np.ndarray:
    with ProcessPoolExecutor(JOBS) as pool:
        return np.stack(list(pool.map(fn, jobs)))


def _filter_tables(probs: str, post: PostParams, stems: list[str], jobs_by_stem: dict,
                   agree: bool) -> dict:
    """{stem: (stem, features, label, ious, exp_iou)} of the instances of `post`; with `agree` the
    AGREE_FEATURES columns (from AGREE_MAPS) follow the FEATURES ones."""
    key = ("tables", probs, post.to_dict(), stems, AGREE_MAPS if agree else None)

    def compute():
        jobs = [(probs, post, jobs_by_stem[s][0], s, jobs_by_stem[s][1]) for s in stems]
        with ProcessPoolExecutor(JOBS) as pool:
            if agree:
                rows = pool.map(_agree_table, [j + (AGREE_MAPS,) for j in jobs], chunksize=4)
            else:
                rows = pool.map(_image_table, jobs, chunksize=4)
            return {t[0]: t for t in rows}
    return _cached(key, compute)


def fit_filter(tables: dict, train: list[str], cols: list[int], folds: int = 5) -> tuple:
    """Classifier fitted on the instances of `train`, and the drop threshold that maximises the
    PQ of `train` on 5-fold out-of-fold scores by image (the sweep of fp_filter.py)."""
    tr = [s for s in train if len(tables[s][2])]
    X = {s: tables[s][1][:, cols] for s in tr}
    idx = list(range(len(tr)))
    random.Random(0).shuffle(idx)
    oof = {}
    for chunk in np.array_split(idx, folds):
        te = {tr[i] for i in chunk}
        fit = [s for s in tr if s not in te]
        model = make_model("clf").fit(np.concatenate([X[s] for s in fit]),
                                      np.concatenate([tables[s][2] for s in fit]))
        for s in te:
            oof[s] = predict(model, "clf", X[s])

    def pq_at(th):
        return pq(sum(image_stats(tables[s][3], oof[s] >= th if s in oof else np.ones(len(tables[s][2]), bool))
                      for s in train))

    best = (pq_at(-1.0), -1.0)
    for th in np.arange(0.05, 0.75, 0.025):
        v = pq_at(th)
        if v > best[0]:
            best = (v, float(th))
    model = make_model("clf").fit(np.concatenate([X[s] for s in tr]), np.concatenate([tables[s][2] for s in tr]))
    return model, best[1]


def evaluate_crosstuned_filter(probs: str, mode: str, fset: str, stems: list[str], jobs_by_stem: dict,
                               repeats: int = 3) -> np.ndarray:
    """evaluate_crosstuned with an instance filter on the features `fset` also learned on the tuning
    half: grid params, classifier and threshold all come from half A, half B is scored."""
    if fset not in FEATURE_SETS:
        raise SystemExit(f"unknown feature set {fset!r}; choose from {list(FEATURE_SETS)}")
    agree = fset == "base+agree"
    if agree and len(AGREE_MAPS) != 4:
        raise SystemExit("filter:base+agree needs --agree-maps with the 4 other models' val maps")
    cols = [(FEATURES + AGREE_FEATURES).index(n) for n in FEATURE_SETS[fset]]
    combos, per_combo = _grid_stats(probs, mode, stems, jobs_by_stem)
    out = np.zeros((len(stems), 4))
    tables_of: dict[int, dict] = {}
    report = []
    for tune_on in _splits(len(stems), repeats):
        best = int(np.argmax([pq(per_combo[c][tune_on].sum(0)) for c in range(len(combos))]))
        if best not in tables_of:
            tables_of[best] = _filter_tables(probs, combos[best], stems, jobs_by_stem, agree)
        tables = tables_of[best]
        model, th = fit_filter(tables, [s for s, t in zip(stems, tune_on) if t], cols)
        for i in np.nonzero(~tune_on)[0]:
            t = tables[stems[i]]
            keep = predict(model, "clf", t[1][:, cols]) >= th if len(t[2]) else np.ones(0, bool)
            out[i] += image_stats(t[3], keep)
        report.append((best, round(th, 3)))
    print(f"  {probs} {mode} filter:{fset}: (combo, threshold) per split {report}", file=sys.stderr, flush=True)
    return out / repeats


def evaluate(spec: str, stems: list[str], jobs_by_stem: dict, folds: int = 5) -> np.ndarray:
    """Per-image stats (n_images, 4) of one spec, in the order of `stems`."""
    name, rest = spec.split("=", 1)
    parts = rest.split(",")
    if parts[1] == "tune:det":
        return evaluate_crosstuned_det(parts[0], stems, jobs_by_stem)
    if parts[1] == "tune:fuse":
        assert len(parts) > 2 and parts[2].startswith("dets:"), "tune:fuse needs dets:<dir>"
        return evaluate_crosstuned_fuse(parts[0], parts[2][5:], stems, jobs_by_stem)
    if parts[1].startswith("tune:"):
        if len(parts) > 2 and parts[2].startswith("filter:"):
            return evaluate_crosstuned_filter(parts[0], parts[1][5:], parts[2][7:], stems, jobs_by_stem)
        return evaluate_crosstuned(parts[0], parts[1][5:], stems, jobs_by_stem)
    probs, post = parts[0], PostParams(**json.loads(Path(parts[1]).read_text()))
    filt = pickle.loads(Path(parts[2]).read_bytes()) if len(parts) > 2 else None
    jobs = [(probs, post, jobs_by_stem[s][0], s, jobs_by_stem[s][1]) for s in stems]
    with ProcessPoolExecutor(JOBS) as pool:
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
    ap.add_argument("--jobs", type=int, help="worker processes (default: all cores)")
    ap.add_argument("--agree-maps", help="comma-separated val maps of the 4 other models (filter:base+agree)")
    ap.add_argument("--cache", type=Path, help="directory for cached grid stats and filter tables")
    args = ap.parse_args()
    global JOBS, CACHE, AGREE_MAPS
    JOBS, CACHE = args.jobs, args.cache
    AGREE_MAPS = args.agree_maps.split(",") if args.agree_maps else []

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
