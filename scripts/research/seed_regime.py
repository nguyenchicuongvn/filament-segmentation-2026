"""SEED-REGIME, gate G2 (PLAN section 0.6): does the stricter seed regime of the 10-model test mean hurt PQ?

Measurement only; nothing in src/, make_submission.py or the configs is changed.

The final pipeline (N1 + N9' pool, hyst thresh 0.5 / seed 0.99 / min_area 200, oriented grouping,
FP filter at 0.275) was tuned on OOF maps that average 2 models per image, and runs on test maps that
average 10. This script
  (a) reproduces the OOF PQ of the final (fp_filter.py protocol: filter scores cross-validated by image);
  (b) re-measures the seed shift without labels: components (p > 0.5, area >= 200) and seeded components
      (peak > 0.99) per image on the OOF pool, the 10-model test pool and the 5 test pair maps;
  (c) emulates the stricter regime on the 707 OOF images and reports paired bootstrap deltas (2000
      resamples over images, generator seed 0, as scripts/compare.py):
        E1  a scalar seed raised (254, 255) and, because the pool saturates at 255, continued with
            "at least k pixels above the seed" until the measured share of seeded components is lost;
        E2  the seed decision taken on a map that averages more OOF models of the same image
            (extent, grouping and filter features stay on the N1 + N9' pool): M3 = N1/2 + (N9' + N11)/4
            (N11 = N9' recipe, seed 2: same family), M4 = (N1 + N2 + N9' + N11)/4 and the 3-family pool
            P3 (both confounded by N2), and E2_M4_match = the measured share removed in M4-peak order;
        E3  inside E2's regime, the vote rule over single-pair-like maps versus the averaged seed;
  (d) describes the components the test mean rejects and the 3/5 pair vote accepts, and the precision
      of their OOF analogues against the PQ break-even P(match) * IoU > PQ / 2.

Every variant only changes WHICH components of the pooled p > 0.5 map are kept. It is run through the
unchanged filseg.dl.instances_lowres by rewriting the peak of each component (kept: one pixel set to
1.0 if needed; dropped: clipped to 0.98), so grouping, min_area and relabelling are the real code.
Filter features are always computed on the original pooled map.

Filter protocols for a changed instance set (threshold frozen at 0.275):
  transfer  fold models fitted on the BASELINE instances of the other image folds score the variant's
            instances (what happens on test: the filter was fitted in the OOF regime);
  refit     fold models refitted on the variant's own instances (fp_filter.py run on the variant);
  nofilt    no filter.

Usage (from the repo root; caches go to --cache; one OpenMP thread keeps the filter fits from stalling
on a busy machine and does not change their result):
  OMP_NUM_THREADS=1 .venv/bin/python -u scripts/research/seed_regime.py --cache <dir> --jobs 4 \
      > docs/logs/final/seed_regime.txt
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filseg.classical import fit_disk  # noqa: E402
from filseg.data import encode_masks, load_gray, load_train, test_paths, with_maps  # noqa: E402
from filseg.dl import PostParams, instances_lowres, load_maps, upsample_labels  # noqa: E402
from filseg.instfilter import instance_features  # noqa: E402
from filseg.metric import IOU_THRESHOLD  # noqa: E402
from fp_filter import _image_table, make_model, predict  # noqa: E402

OOF = ROOT / "outputs/ens/abl/n1n9p_oof"
TEST = ROOT / "outputs/ens/abl/n1n9p_test"
OOF3 = ROOT / "outputs/ens/n1n2n9p_oof"
COLAB = ROOT / "outputs/colab"
FINAL_MAPS = ROOT / "outputs/final/maps"
POST = ROOT / "configs/post_ens_n1n2n9p_hyst_orient.json"
FILTER = ROOT / "configs/fpfilter_ens_n1n9p_orient.pkl"
SUBMITTED = ROOT / "outputs/submission_ens_n1n9p_orient.csv"
# OOF models of one image: N1, N9', N11 (= N9' recipe, --seed 2), N2 (EffNet-B4 crop 2048, other family)
OOF_MODELS = {"n1": "effb4_s1024", "n9": "effv2s_c512", "n11": "effv2s_c512_s2", "n2": "effb4_c2048"}
SEED_U8 = 253          # peak > 0.99 on a uint8 map  <=>  peak >= 253
MIN_AREA, RES = 200, 2048


def _u8_or_up(path: Path) -> np.ndarray:
    """A cached uint8 map as float32 at 2048 px (1024 px maps upsampled as ensemble_probs.py does)."""
    m = np.load(path)
    if m.shape[0] == RES:
        return m
    return cv2.resize(m.astype(np.float32), (RES, RES), interpolation=cv2.INTER_LINEAR)


def _components(pool_u8: np.ndarray):
    """Components of p > 0.5 with area >= MIN_AREA, labelled 1..n in cv2 order; areas, centroids."""
    n, lab, stats, cent = cv2.connectedComponentsWithStats((pool_u8 > 127).astype(np.uint8), 8)
    keep = stats[:, cv2.CC_STAT_AREA] >= MIN_AREA
    keep[0] = False
    lut = np.zeros(n, np.int32)
    lut[keep] = np.arange(1, int(keep.sum()) + 1)
    return lut[lab], stats[keep, cv2.CC_STAT_AREA].astype(np.int64), cent[keep]


def _fg(lab: np.ndarray):
    idx = np.flatnonzero(lab)
    return idx, lab.ravel()[idx]


def _peaks_at(l: np.ndarray, n: int, vals: np.ndarray) -> np.ndarray:
    """Per-component maximum of `vals` (values at the foreground pixels whose labels are `l`)."""
    pk = np.zeros(n + 1, np.float32)
    np.maximum.at(pk, l, vals.astype(np.float32))
    return pk[1:]


def _own_counts(m: np.ndarray) -> tuple[int, int]:
    """(components, seeded components) of a map's own p > 0.5 mask, map rounded to uint8 first."""
    u8 = np.round(m).astype(np.uint8)
    lab, area, _ = _components(u8)
    idx, l = _fg(lab)
    return len(area), int((_peaks_at(l, len(area), u8.ravel()[idx]) >= SEED_U8).sum())


def _disk_xy(gray_path: Path, cent: np.ndarray) -> np.ndarray:
    gray8 = cv2.resize(load_gray(gray_path), (RES, RES), interpolation=cv2.INTER_AREA)
    dcx, dcy, dr = fit_disk(gray8)
    return np.column_stack([(cent[:, 0] - dcx) / dr, (cent[:, 1] - dcy) / dr]) if len(cent) else np.zeros((0, 2))


def comps_oof(args):
    """Per-component table of one OOF image: peaks under every candidate seed map + standalone match."""
    stem, gray_path, gts, model_paths = args
    pool = np.load(OOF / f"{stem}.npy")
    lab, area, cent = _components(pool)
    n = len(area)
    idx, l = _fg(lab)
    m = {k: _u8_or_up(Path(p)).ravel()[idx].astype(np.float32) for k, p in model_paths.items()}
    seed_maps = {
        "A": 0.5 * m["n1"] + 0.5 * m["n9"],            # the pool itself (pair N1 + N9')
        "B": 0.5 * m["n1"] + 0.5 * m["n11"],           # pair-like: N1 + second-seed N9'
        "C": 0.5 * m["n2"] + 0.5 * m["n9"],            # pair-like, N2 in place of N1 (other family)
        "D": 0.5 * m["n2"] + 0.5 * m["n11"],
        "M3": 0.5 * m["n1"] + 0.25 * m["n9"] + 0.25 * m["n11"],                   # mean(A, B)
        "M4": 0.25 * (m["n1"] + m["n2"] + m["n9"] + m["n11"]),                    # mean(A, D) = mean(B, C)
        "P3": np.load(OOF3 / f"{stem}.npy").ravel()[idx].astype(np.float32),      # 3-family OOF pool
        **m,
    }
    pool_fg = pool.ravel()[idx]
    out = {"stem": stem, "area": area, "cent": cent, "n_entries": len(gts),
           "pool_peak": _peaks_at(l, n, pool_fg),
           "pool_check": int((np.round(seed_maps["A"]).astype(np.uint8) != pool_fg).sum()),
           "peak": {k: _peaks_at(l, n, v) for k, v in seed_maps.items()},
           "xy": _disk_xy(gray_path, cent)}
    # standalone: each component as its own mask against every annotator entry
    rles = encode_masks(lab)
    nmatch, siou = np.zeros(n), np.zeros(n)
    for gt in gts:
        if gt and rles:
            iou = np.asarray(mask_util.iou(rles, gt, [0] * len(gt))).T     # (n_gt, n_comp)
            hit = iou > IOU_THRESHOLD
            nmatch += hit.any(0)
            siou += np.where(hit, iou, 0).max(0)
    out["nmatch"], out["siou"] = nmatch, siou
    return out


def seed_pixels(stem: str) -> np.ndarray:
    """Per component of the OOF pool: number of pixels above the seed (uint8 >= 253)."""
    pool = np.load(OOF / f"{stem}.npy")
    lab, area, _ = _components(pool)
    idx, l = _fg(lab)
    return np.bincount(l[pool.ravel()[idx] >= SEED_U8], minlength=len(area) + 1)[1:]


def comps_test(args):
    """Per-component table of one test image: pooled peak and the peak in each of the 5 pair maps."""
    stem, gray_path = args
    pool = np.load(TEST / f"{stem}.npy")
    lab, area, cent = _components(pool)
    n = len(area)
    idx, l = _fg(lab)
    pairs = [np.float32(0.5) * _u8_or_up(FINAL_MAPS / f"effb4_s1024_f{k}" / f"{stem}.npy")
             + np.float32(0.5) * np.load(FINAL_MAPS / f"effv2s_c512_f{k}" / f"{stem}.npy") for k in range(5)]
    mean10 = sum(pairs) / 5
    return {"stem": stem, "area": area, "cent": cent, "xy": _disk_xy(gray_path, cent),
            "pool_peak": _peaks_at(l, n, pool.ravel()[idx]),
            "pair_peak": np.stack([_peaks_at(l, n, p.ravel()[idx]) for p in pairs], 1) if n else np.zeros((0, 5), np.float32),
            "mean10_peak": _peaks_at(l, n, mean10.ravel()[idx]),
            "mean10_diff": int((np.round(mean10).astype(np.uint8) != pool).sum()),
            "mean10_maxdiff": float(np.abs(mean10 - pool).max()),
            "own_pair": [_own_counts(p) for p in pairs]}


def _table(inst: np.ndarray, prob: np.ndarray, gray: np.ndarray, gts: list):
    """fp_filter._image_table for given instance labels: features, clf label, IoU matrices."""
    feats = instance_features(inst, prob, gray, None)
    rles = encode_masks(upsample_labels(inst))
    ious = []
    for gt in gts:
        if gt and rles:
            ious.append(np.asarray(mask_util.iou(rles, gt, [0] * len(gt))).T)
        else:
            ious.append(np.zeros((len(gt), len(rles))))
    label = np.zeros(len(rles), bool)
    for mm in ious:
        if mm.size:
            label |= (mm > IOU_THRESHOLD).any(0)
    return feats, label, ious


def eval_image(args):
    """Tables of one OOF image for each distinct keep set (a bool per component of _components) that
    differs from the baseline keep set (peak > seed_thresh); baseline ones are not recomputed."""
    stem, gray_path, gts, keeps, post = args
    prob, _ = load_maps(OOF, stem)
    lab, area, _ = _components(np.load(OOF / f"{stem}.npy"))
    n = len(area)
    idx, l = _fg(lab)
    peak = _peaks_at(l, n, prob.ravel()[idx])
    seeded = peak > post.seed_thresh
    gray = None
    done: dict[bytes, tuple] = {}
    out = {}
    for name, keep in keeps.items():
        keep = np.asarray(keep, bool)
        if np.array_equal(keep, seeded):
            continue
        key = keep.tobytes()
        if key not in done:
            if gray is None:
                gray = load_gray(gray_path)
            p2 = prob.copy()
            for k in np.flatnonzero(keep != seeded):
                mk = lab == k + 1
                if keep[k]:
                    ys, xs = np.nonzero(mk)
                    j = int(np.argmax(prob[ys, xs]))
                    p2[ys[j], xs[j]] = 1.0
                else:
                    p2[mk] = np.minimum(p2[mk], 0.98)
            done[key] = _table(instances_lowres(p2, post, None), prob, gray, gts)
        out[name] = done[key]
    return stem, out


def check_rewrite(args):
    """The keep-set rewrite against the real code path: seed 254/255 via PostParams.seed_thresh."""
    stem, post = args
    from dataclasses import replace
    prob, _ = load_maps(OOF, stem)
    lab, area, _ = _components(np.load(OOF / f"{stem}.npy"))
    idx, l = _fg(lab)
    peak = _peaks_at(l, len(area), prob.ravel()[idx])
    p2 = prob.copy()
    for k in np.flatnonzero((peak > post.seed_thresh) & ~(peak > 253.5 / 255)):
        mk = lab == k + 1
        p2[mk] = np.minimum(p2[mk], 0.98)
    return bool(np.array_equal(instances_lowres(p2, post, None),
                               instances_lowres(prob, replace(post, seed_thresh=253.5 / 255), None)))


def image_stats(ious: list[np.ndarray], keep: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    for m in ious:
        m = m[:, keep]
        hit = m > IOU_THRESHOLD
        out += [m[hit].sum(), hit.sum(), (hit.sum(0) == 0).sum(), (hit.sum(1) == 0).sum()]
    return out


def pq(s: np.ndarray) -> float:
    den = s[1] + 0.5 * s[2] + 0.5 * s[3]
    return s[0] / den if den else 0.0


def _cache(path: Path, compute):
    if path.exists():
        return pickle.loads(path.read_bytes())
    out = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(out))
    tmp.replace(path)
    return out


def _mse(x) -> str:
    x = np.asarray(x, float)
    return f"{x.mean():6.3f} +- {x.std(ddof=1) / np.sqrt(len(x)):.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    t0 = time.time()
    post = PostParams(**json.loads(POST.read_text()))
    filt = pickle.loads(FILTER.read_bytes())
    TH = filt["threshold"]
    assert post.to_dict() == filt["post"] and post.min_area == MIN_AREA and post.thresh == 0.5
    print(f"post {post.to_dict()}\nfilter threshold {TH} kind {filt.get('kind')} features {len(filt['features'])}")

    by_img: dict[str, tuple[Path, list]] = {}
    for it in with_maps(load_train(), OOF):
        by_img.setdefault(it.image_id, (it.path, []))[1].append(it.gt_rles)
    stems = sorted(by_img)
    n_img = len(stems)

    # ------------------------------------------------------------------ (a) baseline, fp_filter.py path
    def base_tables():
        jobs = [(str(OOF), post, p, s, gts) for s, (p, gts) in sorted(by_img.items())]
        with ProcessPoolExecutor(args.jobs) as pool:
            return list(pool.map(_image_table, jobs, chunksize=4))
    tabs = _cache(args.cache / "base_tables.pkl", base_tables)
    base = {t[0]: (t[1], t[2], t[3]) for t in tabs}

    idx = list(range(n_img))
    random.Random(0).shuffle(idx)
    fold_of = np.empty(n_img, int)
    for f, chunk in enumerate(np.array_split(idx, 5)):
        fold_of[chunk] = f

    def stack(tb):
        X = np.concatenate([tb[s][0] for s in stems])
        y = np.concatenate([tb[s][1] for s in stems])
        owner = np.repeat(np.arange(n_img), [len(tb[s][1]) for s in stems])
        return X, y, owner

    Xb, yb, ob = stack(base)
    fold_models = [make_model("clf").fit(Xb[fold_of[ob] != f], yb[fold_of[ob] != f]) for f in range(5)]

    def scores(tb, mode):
        X, y, owner = stack(tb)
        out = np.zeros(len(y))
        for f in range(5):
            te = fold_of[owner] == f
            if not te.any():
                continue
            mdl = fold_models[f] if mode == "transfer" else make_model("clf").fit(X[~te], y[~te])
            out[te] = predict(mdl, "clf", X[te])
        return out, owner

    def per_image(tb, mode):
        """(n_img, 4) stats and instances kept per image."""
        if mode == "nofilt":
            keep_all = [np.ones(len(tb[s][1]), bool) for s in stems]
        else:
            sc, owner = scores(tb, mode)
            keep_all = [sc[owner == i] >= TH for i in range(n_img)]
        st = np.stack([image_stats(tb[s][2], k) for s, k in zip(stems, keep_all)])
        return st, np.array([k.sum() for k in keep_all])

    boot = np.random.default_rng(0).integers(0, n_img, size=(args.boot, n_img))

    def delta(s, ref):
        d = np.array([pq(s[b].sum(0)) - pq(ref[b].sum(0)) for b in boot])
        return (f"{pq(s.sum(0)) - pq(ref.sum(0)):+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
                f" P>0 {np.mean(d > 0):.2f}")

    base_stats = {m: per_image(base, m) for m in ("transfer", "refit", "nofilt")}
    print("\n(a) baseline OOF PQ of the final pipeline (fp_filter.py protocol)")
    for m, (st, ni) in base_stats.items():
        tot = st.sum(0)
        print(f"  {m:8s} PQ {pq(tot):.4f}  TP {tot[1]:.0f} FP {tot[2]:.0f} FN {tot[3]:.0f}  instances/img {ni.mean():.3f}")
    print("  expected (docs/logs/final/fpfilter_n1n9p.txt): 0.4382 TP 4993 FP 2253 FN 3206; no filter 0.4371")
    assert round(pq(base_stats["refit"][0].sum(0)), 4) == 0.4382, "baseline not reproduced"
    PQ0 = pq(base_stats["refit"][0].sum(0))

    # ------------------------------------------------------------------ component tables
    def oof_comps():
        where = {k: {p.stem: str(p) for f in range(5) for p in (COLAB / f"{d}_f{f}" / "probs_val").glob("*.npy")
                     if not p.name.startswith(".")} for k, d in OOF_MODELS.items()}
        jobs = [(s, by_img[s][0], by_img[s][1], {k: where[k][s] for k in OOF_MODELS}) for s in stems]
        with ProcessPoolExecutor(args.jobs) as pool:
            return list(pool.map(comps_oof, jobs, chunksize=2))

    def test_comps():
        jobs = [(p.stem, p) for p in test_paths()]
        with ProcessPoolExecutor(args.jobs) as pool:
            return list(pool.map(comps_test, jobs, chunksize=2))

    co = _cache(args.cache / "comps_oof.pkl", oof_comps)
    ct = _cache(args.cache / "comps_test.pkl", test_comps)
    assert [c["stem"] for c in co] == stems
    print(f"\nOOF pool rebuilt from per-model maps (round(0.5 N1 + 0.5 N9')): "
          f"{sum(c['pool_check'] for c in co)} differing pixels over {n_img} images")
    print(f"test: mean of the 5 MPS pair maps vs the GPU pool {TEST.name}: "
          f"{np.mean([c['mean10_diff'] for c in ct]):.0f} px/img differ after rounding, "
          f"max |diff| {max(c['mean10_maxdiff'] for c in ct):.2f} / 255")

    # ------------------------------------------------------------------ (b) the shift, label-free
    r8 = lambda x: np.round(x)  # noqa: E731  (peak of the rounded map = rounded peak)
    o_n = np.array([len(c["area"]) for c in co])
    o_s = np.array([(c["pool_peak"] >= SEED_U8).sum() for c in co])
    t_n = np.array([len(c["area"]) for c in ct])
    t_s = np.array([(c["pool_peak"] >= SEED_U8).sum() for c in ct])
    sub = pd.read_csv(SUBMITTED)
    t_inst = sub.filament_id.str.rsplit("_", n=1).str[0].value_counts().reindex([c["stem"] for c in ct]).fillna(0).values
    o_inst = base_stats["refit"][1]
    print("\n(b) components per image (p > 0.5, area >= 200) and seeded (peak > 0.99); mean +- SE over images")
    print(f"  {'map':34s} {'n':>4s} {'components':>16s} {'seeded':>16s} {'share':>7s} {'final inst':>16s}")
    print(f"  {'OOF pool n1n9p_oof (2 models)':34s} {n_img:4d} {_mse(o_n):>16s} {_mse(o_s):>16s} {o_s.sum() / o_n.sum():7.3f} {_mse(o_inst):>16s}")
    print(f"  {'test pool n1n9p_test (10 models)':34s} {len(ct):4d} {_mse(t_n):>16s} {_mse(t_s):>16s} {t_s.sum() / t_n.sum():7.3f} {_mse(t_inst):>16s}")
    pair_n = np.array([[k[0] for k in c["own_pair"]] for c in ct])
    pair_s = np.array([[k[1] for k in c["own_pair"]] for c in ct])
    for k in range(5):
        print(f"  {f'test pair {k} (N1 f{k} + N9p f{k}), own mask':34s} {len(ct):4d} {_mse(pair_n[:, k]):>16s} "
              f"{_mse(pair_s[:, k]):>16s} {pair_s[:, k].sum() / pair_n[:, k].sum():7.3f}")
    print(f"  {'test pairs, mean of 5 per image':34s} {len(ct):4d} {_mse(pair_n.mean(1)):>16s} {_mse(pair_s.mean(1)):>16s} "
          f"{pair_s.sum() / pair_n.sum():7.3f}")
    # pooled-mask components seeded in each pair map / by the vote
    pp = [r8(c["pair_peak"]) >= SEED_U8 for c in ct]
    votes = [p.sum(1) for p in pp]
    for k in range(5):
        print(f"  {f'test pooled comps seeded in pair {k}':34s} {len(ct):4d} {'':16s} {_mse([p[:, k].sum() for p in pp]):>16s}")
    for need in (1, 2, 3, 4, 5):
        print(f"  {f'test pooled comps, vote >= {need}/5':34s} {len(ct):4d} {'':16s} {_mse([(v >= need).sum() for v in votes]):>16s}")
    se = lambda a, b: np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))  # noqa: E731
    share_cross = 1 - t_s.mean() / o_s.mean()
    share_within = 1 - t_s.sum() / pair_s.mean(1).sum()
    print(f"  seeded/img test - OOF: {t_s.mean() - o_s.mean():+.3f} +- {se(t_s, o_s):.3f} (z {(t_s.mean() - o_s.mean()) / se(t_s, o_s):+.2f});"
          f" components/img test - OOF: {t_n.mean() - o_n.mean():+.3f} +- {se(t_n, o_n):.3f}")
    print(f"  seeded/img single test pair (mean of 5) - OOF: {pair_s.mean(1).mean() - o_s.mean():+.3f} +- {se(pair_s.mean(1), o_s):.3f};"
          f" components/img: {pair_n.mean(1).mean() - o_n.mean():+.3f} +- {se(pair_n.mean(1), o_n):.3f}")
    d = pair_s.mean(1) - t_s
    print(f"  within test, paired: seeded in a single pair (own mask, mean of 5) - seeded in the 10-model pool:"
          f" {d.mean():+.3f} +- {d.std(ddof=1) / np.sqrt(len(d)):.3f} per image")
    print(f"  share of seeded components lost: test pool vs OOF {share_cross:.3%} (cross-set);"
          f" test pool vs its own single pairs {share_within:.3%} (within test)")
    print(f"  final instances/img: OOF {o_inst.mean():.3f}, submitted test {t_inst.mean():.3f} "
          f"({t_inst.mean() / o_inst.mean() - 1:+.1%}); diff {t_inst.mean() - o_inst.mean():+.3f} +- {se(t_inst, o_inst):.3f}")
    # ------------------------------------------------------------------ (c) keep sets
    pk = lambda c, k: r8(c["peak"][k])  # noqa: E731
    seeded0 = [c["pool_peak"] >= SEED_U8 for c in co]
    n_seed0 = sum(s.sum() for s in seeded0)
    variants: dict[str, list[np.ndarray]] = {"base": seeded0}
    variants["E1_u8_254"] = [c["pool_peak"] >= 254 for c in co]
    variants["E1_u8_255"] = [c["pool_peak"] >= 255 for c in co]
    # The pool saturates: most seeded components peak at exactly 255, so the scalar seed cannot go past
    # "peak >= 255". To lose the measured share the ladder continues with the number of pixels above the
    # seed (>= 253): "at least k seed pixels" (k = 1 is the baseline).
    def seed_px():
        with ProcessPoolExecutor(args.jobs) as pool:
            return list(pool.map(seed_pixels, stems, chunksize=8))
    spx = _cache(args.cache / "seed_px.pkl", seed_px)
    assert all(((n > 0) == s).all() for n, s in zip(spx, seeded0))
    lost_at = lambda k: 1 - sum(int((n >= k).sum()) for n in spx) / n_seed0  # noqa: E731
    ks = np.arange(1, 400)
    lost = np.array([lost_at(k) for k in ks])
    print(f"\n  scalar seed: peak >= 254 loses {1 - sum(int(v.sum()) for v in variants['E1_u8_254']) / n_seed0:.2%}, "
          f"peak >= 255 (the maximum) loses {1 - sum(int(v.sum()) for v in variants['E1_u8_255']) / n_seed0:.2%} "
          f"of the seeded components; target {share_cross:.2%} (cross-set) / {share_within:.2%} (within test)")
    k_hi = int(ks[np.argmax(lost >= share_cross)])          # first k losing at least the cross-set share
    k_w = int(ks[np.argmax(lost >= share_within)])
    for k in sorted({k_hi - 1, k_hi, k_w}):
        variants[f"E1_seedpx>={k}"] = [n >= k for n in spx]
        print(f"  seed pixels >= {k}: loses {lost_at(k):.2%}")
    E1_AREA = [n for n in variants if n.startswith("E1_seedpx")]
    # the same share, but removing the components the extra averaging itself ranks lowest: baseline-seeded
    # components ordered by their peak in the 4-model mean M4 (float, not rounded)
    m4 = np.concatenate([c["peak"]["M4"][s] for c, s in zip(co, seeded0)])
    t4 = float(np.quantile(m4, share_cross))
    assert t4 < 255, "M4 saturated too: no matched threshold"
    variants["E2_M4_match"] = [s & (c["peak"]["M4"] > t4) for c, s in zip(co, seeded0)]
    print(f"  E2_M4_match: baseline-seeded with M4 peak > {t4:.3f} / 255 (M4 peak == 255 for {np.mean(m4 >= 255):.1%} of them)")
    for k in ("P3", "M3", "M4"):
        variants[f"E2_{k}"] = [pk(c, k) >= SEED_U8 for c in co]
    sA = [pk(c, "A") >= SEED_U8 for c in co]
    sB = [pk(c, "B") >= SEED_U8 for c in co]
    sC = [pk(c, "C") >= SEED_U8 for c in co]
    sD = [pk(c, "D") >= SEED_U8 for c in co]
    assert all((a == s).all() for a, s in zip(sA, seeded0))
    variants["E3_M3_any(A,B)"] = [a | b for a, b in zip(sA, sB)]
    variants["E3_M3_all(A,B)"] = [a & b for a, b in zip(sA, sB)]
    variants["E3_M4_any(A,D)"] = [a | d_ for a, d_ in zip(sA, sD)]
    variants["E3_M4_all(A,D)"] = [a & d_ for a, d_ in zip(sA, sD)]
    v4 = [a.astype(int) + b + c_ + d_ for a, b, c_, d_ in zip(sA, sB, sC, sD)]
    variants["E3_M4_vote2of4"] = [v >= 2 for v in v4]
    variants["E3_M4_vote3of4"] = [v >= 3 for v in v4]

    names = list(variants)
    sums = {n: sum(int(v.sum()) for v in variants[n]) for n in names}
    for n in names[1:]:
        ch = sum(int(not np.array_equal(v, s)) for v, s in zip(variants[n], seeded0))
        print(f"  keep set {n:18s} {sums[n]:5d} comps ({sums[n] / n_seed0 - 1:+.2%} vs base), {ch} images changed", flush=True)

    with ProcessPoolExecutor(args.jobs) as pool:
        changed = [i for i, (a, b) in enumerate(zip(variants["E1_u8_254"], seeded0)) if not np.array_equal(a, b)][:12]
        ok = list(pool.map(check_rewrite, [(stems[i], post) for i in changed]))
    print(f"\n(c) keep-set rewrite vs PostParams(seed_thresh=253.5/255) on {len(ok)} changed images: "
          f"{sum(ok)} identical label images", flush=True)
    assert all(ok)

    tb = {"base": base}
    groups = {"E1": [n for n in names if n.startswith("E1")], "E2": [n for n in names if n.startswith("E2")],
              "E3": [n for n in names if n.startswith("E3")]}

    def eval_group(g):
        def run():
            jobs = [(s, by_img[s][0], by_img[s][1], {n: variants[n][i] for n in groups[g]}, post)
                    for i, s in enumerate(stems)]
            with ProcessPoolExecutor(args.jobs) as pool:
                return dict(pool.map(eval_image, jobs, chunksize=2))
        sig = "_".join(f"{n}{sums[n]}" for n in groups[g])
        import hashlib
        ev = _cache(args.cache / f"eval_{g}_{hashlib.sha1(sig.encode()).hexdigest()[:10]}.pkl", run)
        for n in groups[g]:
            tb[n] = {s: ev[s].get(n, base[s]) for s in stems}
        print(f"  [{g} evaluated, {time.time() - t0:.0f} s]", flush=True)

    res = {"base": base_stats}

    def add(n):
        res[n] = {m: per_image(tb[n], m) for m in ("transfer", "refit", "nofilt")}

    def row(n, ref, modes=("transfer", "refit", "nofilt")):
        kept = sum(int(v.sum()) for v in variants[n])
        lost = sum(int((s & ~v).sum()) for s, v in zip(variants[ref], variants[n]))
        gained = sum(int((v & ~s).sum()) for s, v in zip(variants[ref], variants[n]))
        nref = sum(int(v.sum()) for v in variants[ref])
        print(f"  {n} vs {ref}: seeded comps {kept} ({kept / n_img:.3f}/img), lost {lost} ({lost / nref:.2%}), "
              f"gained {gained} ({gained / nref:.2%}), net {kept / nref - 1:+.2%}")
        for m in modes:
            st, ni = res[n][m]
            tot = st.sum(0)
            print(f"      {m:8s} PQ {pq(tot):.4f} TP {tot[1]:.0f} FP {tot[2]:.0f} FN {tot[3]:.0f} inst/img {ni.mean():.3f}"
                  f"  delta {delta(st, res[ref][m][0])}")

    print(f"    baseline: {n_seed0} seeded comps ({n_seed0 / n_img:.3f}/img); delta = variant - reference, "
          f"paired bootstrap over {n_img} images, {args.boot} resamples, 95% CI")
    print("\n  E1: scalar seed raised on the OOF pool (strict side, same share lost as measured)")
    eval_group("E1")
    for n in ["E1_u8_254", "E1_u8_255", *E1_AREA]:
        add(n)
        row(n, "base")
    print("\n  E2: seed decided on a map averaging more OOF models (extent/grouping/features on the N1 + N9p pool)")
    eval_group("E2")
    for n in ("E2_M3", "E2_M4", "E2_P3", "E2_M4_match"):
        add(n)
        row(n, "base")
    print("\n  E3: vote over pair-like maps versus the averaged seed of the same regime")
    for n, ref in (("E3_M3_any(A,B)", "E2_M3"), ("E3_M3_all(A,B)", "E2_M3"), ("E3_M4_any(A,D)", "E2_M4"),
                   ("E3_M4_all(A,D)", "E2_M4"), ("E3_M4_vote2of4", "E2_M4"), ("E3_M4_vote3of4", "E2_M4")):
        if n not in res:
            if "E3_M3_any(A,B)" not in tb:
                eval_group("E3")
            add(n)
        row(n, ref)
    print("\n  for reference, the vote variants versus the baseline (single pair A)")
    for n in ("E3_M3_any(A,B)", "E3_M3_all(A,B)", "E3_M4_any(A,D)", "E3_M4_all(A,D)", "E3_M4_vote2of4", "E3_M4_vote3of4"):
        row(n, "base", modes=("transfer",))

    # ------------------------------------------------------------------ (d) marginal components
    print(f"\n(d) marginal components. Break-even: keep iff P(match per entry) * IoU > PQ/2 = {PQ0 / 2:.3f}")
    def describe(area, peak, xy):
        r = np.hypot(xy[:, 0], xy[:, 1])
        q = lambda x: "/".join(f"{v:.3g}" for v in np.percentile(x, [25, 50, 75]))  # noqa: E731
        return (f"n {len(area):5d}  area q25/50/75 {q(area)}  pooled peak {q(peak)}  "
                f"disk radius {q(r)}  |x| {q(np.abs(xy[:, 0]))}  r>0.8 {np.mean(r > 0.8):.2f}")
    T = {k: np.concatenate([c[k] for c in ct]) for k in ("area", "pool_peak", "xy")}
    tv = np.concatenate(votes)
    tseed = T["pool_peak"] >= SEED_U8
    print("  test components (pooled mask), label-free:")
    for lbl, m in (("pool seeded (kept now)", tseed), ("pool rejects, vote >= 3/5 (ADDED by the rule)", ~tseed & (tv >= 3)),
                   ("pool seeded, vote < 3/5 (REMOVED by the rule)", tseed & (tv < 3)),
                   ("pool rejects, vote 1-2/5", ~tseed & (tv >= 1) & (tv < 3)), ("pool rejects, vote 0/5", ~tseed & (tv == 0))):
        if m.any():
            print(f"    {lbl:46s} {describe(T['area'][m], T['pool_peak'][m], T['xy'][m])}")
        else:
            print(f"    {lbl:46s} n     0")
    print(f"    added/img {(~tseed & (tv >= 3)).sum() / len(ct):.3f}, removed/img {(tseed & (tv < 3)).sum() / len(ct):.3f}; "
          f"pooled peak of the added: " + ", ".join(f"{int(v)}: {int(n)}" for v, n in
          zip(*np.unique(T["pool_peak"][~tseed & (tv >= 3)], return_counts=True))))
    vt = np.array([(tv == k).sum() for k in range(6)])
    print(f"    vote histogram of all test comps 0..5: {vt.tolist()}; of pool-seeded: "
          f"{[int((tv[tseed] == k).sum()) for k in range(6)]}; of pool-rejected: {[int((tv[~tseed] == k).sum()) for k in range(6)]}")

    O = {k: np.concatenate([c[k] for c in co]) for k in ("area", "pool_peak", "xy", "nmatch", "siou")}
    O["ne"] = np.concatenate([np.full(len(c["area"]), c["n_entries"]) for c in co])
    cat = lambda v: np.concatenate(v)  # noqa: E731
    def prec(m, lbl):
        if not m.any():
            print(f"    {lbl:52s} n     0")
            return
        P = O["nmatch"][m].sum() / O["ne"][m].sum()
        iou = O["siou"][m].sum() / max(O["nmatch"][m].sum(), 1)
        val = O["siou"][m].sum() / O["ne"][m].sum()
        print(f"    {lbl:52s} n {int(m.sum()):5d}  P(match/entry) {P:.3f}  IoU|match {iou:.3f}  P*IoU {val:.3f} "
              f"({'above' if val > PQ0 / 2 else 'BELOW'} {PQ0 / 2:.3f})  any-annotator {np.mean(O['nmatch'][m] > 0):.3f}")
    print("  OOF analogues, each component scored alone against every annotator entry of its image (before grouping/filter):")
    s0 = cat(seeded0)
    prec(s0, "baseline seeded (pool peak >= 253)")
    for lv in (255, 254, 253, 252, 251, 250):
        prec(O["pool_peak"] == lv, f"pool peak == {lv}")
    prec((O["pool_peak"] < 250) & (O["pool_peak"] >= 240), "pool peak 240..249")
    prec(O["pool_peak"] < 240, "pool peak < 240")
    for n in ["E1_u8_255", *E1_AREA, "E2_M3", "E2_M4", "E2_P3", "E2_M4_match"]:
        prec(s0 & ~cat(variants[n]), f"lost by {n} (seeded in base)")
    for n, ref in (("E3_M3_any(A,B)", "E2_M3"), ("E3_M4_any(A,D)", "E2_M4"), ("E3_M4_vote2of4", "E2_M4"),
                   ("E3_M4_vote3of4", "E2_M4"), ("E3_M4_all(A,D)", "E2_M4")):
        a, r_ = cat(variants[n]), cat(variants[ref])
        prec(a & ~r_, f"ADDED by {n} over {ref}")
        prec(~a & r_, f"REMOVED by {n} from {ref}")
    for lbl, m in (("OOF: lost by E2_M4, seeded in base", s0 & ~cat(variants["E2_M4"])),
                   ("OOF: added by vote2of4 over E2_M4", cat(variants["E3_M4_vote2of4"]) & ~cat(variants["E2_M4"]))):
        if m.any():
            print(f"    {lbl:46s} {describe(O['area'][m], O['pool_peak'][m], O['xy'][m])}")
    print(f"\ndone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
