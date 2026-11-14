"""Independent check of gate G2 (and G1) of SEED-REGIME, PLAN section 0.6. Verification only.

Differs from scripts/research/seed_regime.py on purpose:
  * a dropped component is zeroed in the map (not clipped to 0.98) before the unchanged instances_lowres;
  * own PQ bookkeeping; bootstrap generator seed 123 (paired by image, 2000 resamples);
  * a TEST-CALIBRATED emulation: on the 180 test images (label-free), every component seeded in a single
    pair map (the OOF regime) is labelled "lost" when the 10-model pool has no seed inside it. A model
    P(lost | pair-map features) is fitted on test and applied to the OOF pool components (which are
    single-pair components), so the OOF components removed are the kind the 10-model average removes.
    Removal is (i) rank-matched to the measured share and (ii) Bernoulli draws from the propensity.
  * re-evaluates the report's E1 scalar 255 and seed-area >= 163 keep sets for cross-checking.

Usage: OMP_NUM_THREADS=1 .venv/bin/python -u scripts/research/seed_regime_verify.py --cache DIR --jobs 4
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filseg.data import encode_masks, load_gray, load_train, with_maps  # noqa: E402
from filseg.dl import PostParams, instances_lowres, upsample_labels  # noqa: E402
from filseg.instfilter import instance_features  # noqa: E402
from fp_filter import make_model  # noqa: E402

OOF = ROOT / "outputs/ens/abl/n1n9p_oof"
TEST = ROOT / "outputs/ens/abl/n1n9p_test"
PAIRS = [ROOT / f"outputs/final/maps/pair_f{k}" for k in range(5)]
POST = ROOT / "configs/post_ens_n1n2n9p_hyst_orient.json"
FILTER = ROOT / "configs/fpfilter_ens_n1n9p_orient.pkl"
SEED8 = 253


def comps(u8: np.ndarray):
    n, lab, stats, _ = cv2.connectedComponentsWithStats((u8 > 127).astype(np.uint8), 8)
    ok = stats[:, cv2.CC_STAT_AREA] >= 200
    ok[0] = False
    lut = np.zeros(n, np.int32)
    lut[ok] = np.arange(1, int(ok.sum()) + 1)
    return lut[lab], int(ok.sum())


def comp_feats(u8: np.ndarray, lab: np.ndarray, n: int, other: np.ndarray | None = None):
    """Per component: area, peak, seed px, px == 255, mean, frac > 204 (+ peak of `other` inside it)."""
    idx = np.flatnonzero(lab)
    l = lab.ravel()[idx]
    v = u8.ravel()[idx]
    area = np.bincount(l, minlength=n + 1)[1:]
    peak = np.zeros(n + 1)
    np.maximum.at(peak, l, v)
    spx = np.bincount(l[v >= SEED8], minlength=n + 1)[1:]
    n255 = np.bincount(l[v >= 255], minlength=n + 1)[1:]
    mean = np.bincount(l, weights=v, minlength=n + 1)[1:] / np.maximum(area, 1)
    f80 = np.bincount(l[v > 204], minlength=n + 1)[1:] / np.maximum(area, 1)
    X = np.column_stack([area, peak[1:], spx, n255, mean, f80]).astype(float)
    if other is None:
        return X
    opk = np.zeros(n + 1)
    np.maximum.at(opk, l, other.ravel()[idx])
    return X, opk[1:]


def test_image(stem: str):
    pool = np.load(TEST / f"{stem}.npy")
    lab, n = comps(pool)
    Xp = comp_feats(pool, lab, n)
    rows = []
    for k, d in enumerate(PAIRS):
        pm = np.load(d / f"{stem}.npy")
        lab, n = comps(pm)
        X, opk = comp_feats(pm, lab, n, pool)
        rows.append((k, X, opk))
    return stem, Xp, rows


def oof_image(stem: str):
    pool = np.load(OOF / f"{stem}.npy")
    lab, n = comps(pool)
    return stem, comp_feats(pool, lab, n)


def eval_image(args):
    stem, gray_path, gts, keeps, post = args
    u8 = np.load(OOF / f"{stem}.npy")
    prob = u8.astype(np.float32) / 255.0
    lab, n = comps(u8)
    gray = load_gray(gray_path)

    def table(p):
        inst = instances_lowres(p, post, None)
        feats = instance_features(inst, prob, gray, None)
        rles = encode_masks(upsample_labels(inst))
        ious = [np.asarray(mask_util.iou(rles, gt, [0] * len(gt))).T if gt and rles
                else np.zeros((len(gt), len(rles))) for gt in gts]
        y = np.zeros(len(rles), bool)
        for m in ious:
            if m.size:
                y |= (m > 0.5).any(0)
        return feats, y, ious

    out = {"base": table(prob)}
    peak = np.zeros(n + 1)
    np.maximum.at(peak, lab.ravel(), u8.ravel())
    seeded = peak[1:] >= SEED8
    done = {}
    for name, keep in keeps.items():
        keep = np.asarray(keep, bool)
        assert not (keep & ~seeded).any()
        if np.array_equal(keep, seeded):
            continue
        key = keep.tobytes()
        if key not in done:
            drop = np.zeros(n + 1, bool)
            drop[1:] = seeded & ~keep
            p2 = prob.copy()
            p2[drop[lab]] = 0.0
            done[key] = table(p2)
        out[name] = done[key]
    return stem, out


def cache(path: Path, fn):
    if path.exists():
        return pickle.loads(path.read_bytes())
    out = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    post = PostParams(**json.loads(POST.read_text()))
    TH = pickle.loads(FILTER.read_bytes())["threshold"]

    by_img: dict[str, tuple[Path, list]] = {}
    for it in with_maps(load_train(), OOF):
        by_img.setdefault(it.image_id, (it.path, []))[1].append(it.gt_rles)
    stems = sorted(by_img)
    N = len(stems)
    tstems = sorted(p.stem for p in TEST.glob("*.npy") if not p.name.startswith("."))
    print(f"{N} OOF images, {len(tstems)} test images")

    def run(fn, jobs, cs=4):
        with ProcessPoolExecutor(args.jobs) as pool:
            return list(pool.map(fn, jobs, chunksize=cs))

    tt = cache(args.cache / "test.pkl", lambda: run(test_image, tstems))
    oo = cache(args.cache / "oof.pkl", lambda: run(oof_image, stems, 8))
    assert [o[0] for o in oo] == stems

    # ---------------------------------------------------------------- shift, label-free
    o_seed = np.array([(o[1][:, 1] >= SEED8).sum() for o in oo])
    o_n = np.array([len(o[1]) for o in oo])
    t_seed = np.array([(t[1][:, 1] >= SEED8).sum() for t in tt])
    t_n = np.array([len(t[1]) for t in tt])
    share_cross = 1 - t_seed.mean() / o_seed.mean()
    print(f"\nshift: comps/img OOF {o_n.mean():.3f} test {t_n.mean():.3f}; seeded/img OOF {o_seed.mean():.3f} "
          f"test pool {t_seed.mean():.3f}; share lost cross-set {share_cross:.2%}")
    se = np.sqrt(o_seed.var(ddof=1) / N + t_seed.var(ddof=1) / len(tt))
    print(f"  seeded diff {t_seed.mean() - o_seed.mean():+.3f} +- {se:.3f}")

    # ---------------------------------------------------------------- propensity on test
    Xs, ys, gs = [], [], []
    for i, (_, _, rows) in enumerate(tt):
        for k, X, opk in rows:
            s = X[:, 1] >= SEED8
            Xs.append(X[s]); ys.append(opk[s] < SEED8); gs.append(np.full(int(s.sum()), i))
    Xt, yt, gt_ = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)
    share_within = yt.mean()
    print(f"\ntest, single-pair regime: {len(yt)} pair-seeded components ({len(yt) / 5 / len(tt):.3f}/img/pair), "
          f"lost in the 10-model pool {yt.sum()} = {share_within:.2%}")
    names = ["area", "peak", "seedpx", "n255", "mean", "f80"]
    for j, nm in enumerate(names):
        q = lambda a: "/".join(f"{x:.0f}" if nm not in ("f80",) else f"{x:.2f}" for x in np.percentile(a, [10, 25, 50, 75, 90]))  # noqa: E731
        print(f"  {nm:7s} q10/25/50/75/90  lost {q(Xt[yt, j]):32s} kept {q(Xt[~yt, j]):32s} AUC(lower=lost) {roc_auc_score(yt, -Xt[:, j]):.3f}")
    for lo, hi in ((1, 20), (20, 60), (60, 163), (163, 400), (400, 1000), (1000, 10 ** 9)):
        m = (Xt[:, 2] >= lo) & (Xt[:, 2] < hi)
        print(f"  seedpx [{lo},{hi}): n {m.sum():5d} lost rate {yt[m].mean():.3f}")
    m = Xt[:, 2] < 163
    print(f"  rule 'seedpx < 163' on test pairs: flags {m.mean():.2%} of pair-seeded; of the truly lost it catches "
          f"{(m & yt).sum() / yt.sum():.2%}; of the flagged, truly lost {yt[m].mean():.2%}")
    for pkv in (253, 254, 255):
        m = Xt[:, 1] == pkv
        print(f"  pair peak == {pkv}: n {m.sum():5d} ({m.mean():.2%}) lost rate {yt[m].mean():.3f}")

    def hgb():
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=8,
                                              l2_regularization=1.0, random_state=0)
    cv = np.zeros(len(yt))
    fo = np.random.default_rng(1).integers(0, 5, len(tt))
    for f in range(5):
        te = fo[gt_] == f
        cv[te] = hgb().fit(Xt[~te], yt[~te]).predict_proba(Xt[te])[:, 1]
    print(f"  propensity model, CV by test image: AUC {roc_auc_score(yt, cv):.3f} (seedpx alone {roc_auc_score(yt, -Xt[:, 2]):.3f})")
    prop = hgb().fit(Xt, yt)

    # ---------------------------------------------------------------- OOF keep sets
    seeded0 = [o[1][:, 1] >= SEED8 for o in oo]
    n0 = int(sum(s.sum() for s in seeded0))
    Xo = np.concatenate([o[1][s] for o, s in zip(oo, seeded0)])
    po = prop.predict_proba(Xo)[:, 1]
    print(f"\nOOF: {n0} seeded components; propensity expects {po.sum():.0f} lost = {po.mean():.2%} "
          f"(test within {share_within:.2%}, cross-set {share_cross:.2%})")
    for j, nm in enumerate(names):
        print(f"  covariate check {nm:7s} median/mean  OOF seeded {np.median(Xo[:, j]):.2f}/{Xo[:, j].mean():.2f}   test pair-seeded {np.median(Xt[:, j]):.2f}/{Xt[:, j].mean():.2f}")
    print(f"  OOF seeded with seedpx < 163: {(Xo[:, 2] < 163).mean():.2%}; test pair-seeded with seedpx < 163: {(Xt[:, 2] < 163).mean():.2%}")

    def split(flat_keep):
        out, a = [], 0
        for s in seeded0:
            k = s.copy()
            k[s] = flat_keep[a:a + int(s.sum())]
            a += int(s.sum())
            out.append(k)
        return out

    order = np.argsort(-po, kind="stable")
    variants = {}
    for nm, sh in (("prop_rank_cross", share_cross), ("prop_rank_within", share_within)):
        keep = np.ones(n0, bool)
        keep[order[:int(round(sh * n0))]] = False
        variants[nm] = split(keep)
    rng = np.random.default_rng(7)
    for d in range(3):
        variants[f"prop_bern{d}"] = split(rng.random(n0) >= po)
    variants["E1_u8_255"] = split(Xo[:, 1] >= 255)
    variants["E1_seedpx>=163"] = split(Xo[:, 2] >= 163)
    # random removal of the same share: what losing 9.6% of seeded components costs with no selection at all
    variants["random_cross"] = split(rng.random(n0) >= share_cross)
    for nm, v in variants.items():
        k = sum(int(x.sum()) for x in v)
        print(f"  keep set {nm:18s} {k} ({k / n0 - 1:+.2%}), images changed {sum(not np.array_equal(a, b) for a, b in zip(v, seeded0))}")
    lostA = ~np.concatenate([v[s] for v, s in zip(variants['prop_rank_cross'], seeded0)])
    lostB = ~np.concatenate([v[s] for v, s in zip(variants['E1_seedpx>=163'], seeded0)])
    print(f"  overlap of removed sets prop_rank_cross vs seedpx>=163: {(lostA & lostB).sum()} of {lostA.sum()} / {lostB.sum()}", flush=True)

    # ---------------------------------------------------------------- evaluate
    jobs = [(s, by_img[s][0], by_img[s][1], {n: v[i] for n, v in variants.items()}, post) for i, s in enumerate(stems)]
    ev = dict(cache(args.cache / "eval.pkl", lambda: run(eval_image, jobs, 2)))
    print(f"  [evaluated, {time.time() - t0:.0f} s]")

    idx = list(range(N))
    random.Random(0).shuffle(idx)
    fold_of = np.empty(N, int)
    for f, ch in enumerate(np.array_split(idx, 5)):
        fold_of[ch] = f

    def stack(nm):
        tb = [ev[s].get(nm, ev[s]["base"]) for s in stems]
        return tb, np.concatenate([t[0] for t in tb]), np.concatenate([t[1] for t in tb]), \
            np.repeat(np.arange(N), [len(t[1]) for t in tb])

    _, Xb, yb, ob = stack("base")
    fm = [make_model("clf").fit(Xb[fold_of[ob] != f], yb[fold_of[ob] != f]) for f in range(5)]

    def stats(nm, mode):
        tb, X, y, ow = stack(nm)
        if mode == "nofilt":
            sc = np.ones(len(y))
        else:
            sc = np.zeros(len(y))
            for f in range(5):
                te = fold_of[ow] == f
                mdl = fm[f] if mode == "transfer" else make_model("clf").fit(X[~te], y[~te])
                sc[te] = mdl.predict_proba(X[te])[:, 1]
        S = np.zeros((N, 4))
        cnt = np.zeros(N)
        for i, t in enumerate(tb):
            k = sc[ow == i] >= TH
            cnt[i] = k.sum()
            for m in t[2]:
                m = m[:, k]
                hit = m > 0.5
                S[i] += [m[hit].sum(), hit.sum(), (hit.sum(0) == 0).sum(), (hit.sum(1) == 0).sum()]
        return S, cnt

    pq = lambda s: s[0] / (s[1] + 0.5 * s[2] + 0.5 * s[3])  # noqa: E731
    boot = np.random.default_rng(123).integers(0, N, size=(2000, N))
    base = {m: stats("base", m) for m in ("transfer", "refit", "nofilt")}
    print("\nbaseline (expected 0.4382 TP 4993 FP 2253 FN 3206; nofilt 0.4371 TP 5026 FP 2361 FN 3173)")
    for m, (S, c) in base.items():
        t = S.sum(0)
        print(f"  {m:8s} PQ {pq(t):.4f} TP {t[1]:.0f} FP {t[2]:.0f} FN {t[3]:.0f} inst/img {c.mean():.3f} sd {c.std(ddof=1):.3f}")

    print("\ndelta PQ vs baseline, paired bootstrap by image, 2000 resamples, 95% percentile CI")
    for nm in variants:
        for m in ("transfer", "refit", "nofilt"):
            S, c = stats(nm, m)
            R = base[m][0]
            d = np.array([pq(S[b].sum(0)) - pq(R[b].sum(0)) for b in boot])
            t = S.sum(0)
            print(f"  {nm:18s} {m:8s} PQ {pq(t):.4f} TP {t[1]:.0f} FP {t[2]:.0f} FN {t[3]:.0f} inst/img {c.mean():.3f}  "
                  f"delta {pq(t) - pq(R.sum(0)):+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] P<0 {np.mean(d < 0):.3f}")
    # mean of the Bernoulli draws (transfer)
    for m in ("transfer", "nofilt"):
        Ss = [stats(f"prop_bern{d}", m)[0] for d in range(3)]
        R = base[m][0]
        d = np.array([np.mean([pq(S[b].sum(0)) for S in Ss]) - pq(R[b].sum(0)) for b in boot])
        print(f"  prop_bern mean of 3 {m:8s} delta {np.mean([pq(S.sum(0)) for S in Ss]) - pq(R.sum(0)):+.4f} "
              f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")

    # ---------------------------------------------------------------- G1
    print("\nG1 (instance density on test)")
    o_cnt = base["refit"][1]
    se_o = o_cnt.std(ddof=1) / np.sqrt(N)
    for nm, f in (("submitted", "outputs/submission_ens_n1n9p_orient.csv"), ("candidate", "outputs/submission_n1n9p_seedvote.csv")):
        df = pd.read_csv(ROOT / f)
        c = df.iloc[:, 0].astype(str).str.rsplit("_", n=1).str[0].value_counts().reindex(tstems).fillna(0).values
        se_t = c.std(ddof=1) / np.sqrt(len(c))
        sp = np.sqrt(se_o ** 2 + se_t ** 2)
        ref = o_cnt.mean()
        print(f"  {nm}: {len(df)} rows, {c.mean():.4f}/img, SE_test {se_t:.4f}, SE_oof {se_o:.4f}, SE_pooled {sp:.4f}, "
              f"band {ref:.3f} +- {2 * sp:.3f} = [{ref - 2 * sp:.3f}, {ref + 2 * sp:.3f}] inside {abs(c.mean() - ref) <= 2 * sp}; "
              f"|d to OOF| {abs(c.mean() - ref):.3f}, |d to submitted 5.767| {abs(c.mean() - 1038 / 180):.3f}")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
