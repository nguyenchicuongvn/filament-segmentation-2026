"""Does the final-1 FP filter gain from cross-family agreement features?

Same tables, CV split (random.Random(0) over sorted stems, 5 folds by image), model and
threshold sweep as scripts/fp_filter.py; the base run must reproduce its cv_pq. Then the same
with FAMILY_AGREE_FEATURES appended. Paired bootstrap over images of the PQ difference, each
arm at its own best threshold, plus a nested variant (threshold chosen on the other 4 folds).

  python scripts/research/family_agree_filter.py --pool outputs/ens/n1n2n9p_oof \
      --families outputs/ens/n1up_oof,outputs/ens/abl/n2_oof,outputs/ens/n9p_oof \
      --post configs/post_ens_n1n2n9p_hyst_orient.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filseg.data import load_train, with_maps  # noqa: E402
from filseg.dl import PostParams, instances_lowres, load_maps  # noqa: E402
from filseg.instfilter import family_agreement_features  # noqa: E402
from filseg.metric import IOU_THRESHOLD  # noqa: E402

_spec = importlib.util.spec_from_file_location("fp_filter", ROOT / "scripts" / "fp_filter.py")
fpf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fpf)


def _job(args):
    pool, post, path, stem, gts, fams = args
    stem, feats, label, ious, exp_iou = fpf._image_table((pool, post, path, stem, gts))
    prob, group = load_maps(Path(pool), stem)
    inst = instances_lowres(prob, post, group)
    agree = family_agreement_features(inst, [load_maps(Path(d), stem)[0] for d in fams])
    assert len(agree) == len(feats)
    return stem, np.concatenate([feats, agree], 1), label, ious


def per_image(tables, keep):
    """(n_images, 4): sum TP IoU, TP, FP, FN for each image."""
    out = np.zeros((len(tables), 4))
    for i, (stem, _, _, ious) in enumerate(tables):
        for m in ious:
            m = m[:, keep[stem]]
            hit = m > IOU_THRESHOLD
            out[i] += [m[hit].sum(), hit.sum(), (hit.sum(0) == 0).sum(), (hit.sum(1) == 0).sum()]
    return out


def pq(s):
    s = s.sum(0)
    den = s[1] + 0.5 * s[2] + 0.5 * s[3]
    return s[0] / den if den else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--families", required=True)
    ap.add_argument("--post", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    fams = args.families.split(",")
    post = PostParams(**json.loads(args.post.read_text()))
    by_img = {}
    for it in with_maps(load_train(), Path(args.pool)):
        by_img.setdefault(it.image_id, (it.path, []))[1].append(it.gt_rles)
    jobs = [(args.pool, post, p, s, gts, fams) for s, (p, gts) in sorted(by_img.items())]
    with ProcessPoolExecutor(args.jobs) as ex:
        tables = list(ex.map(_job, jobs, chunksize=4))
    nb = len(fpf.FEATURES)
    X = np.concatenate([t[1] for t in tables])
    y = np.concatenate([t[2] for t in tables])
    owner = np.repeat(np.arange(len(tables)), [len(t[2]) for t in tables])
    idx = list(range(len(tables)))
    random.Random(0).shuffle(idx)
    fold_of = np.empty(len(tables), int)
    for f, chunk in enumerate(np.array_split(idx, 5)):
        fold_of[chunk] = f
    grid = np.arange(0.05, 0.75, 0.025)
    results = {}
    for name, cols in [("base", slice(0, nb)), ("agree", slice(None))]:
        oof = np.zeros(len(y))
        for f in range(5):
            te = fold_of[owner] == f
            oof[te] = fpf.make_model("clf").fit(X[~te][:, cols], y[~te]).predict_proba(X[te][:, cols])[:, 1]
        def keep_at(th):
            return {t[0]: oof[owner == i] >= th for i, t in enumerate(tables)}
        stats = {th: per_image(tables, keep_at(th)) for th in grid}
        best = max(grid, key=lambda th: pq(stats[th]))
        # nested: threshold chosen on the other folds' images
        nested = np.zeros((len(tables), 4))
        img_fold = fold_of
        for f in range(5):
            tr = img_fold != f
            th_f = max(grid, key=lambda th: pq(stats[th][tr]))
            nested[img_fold == f] = stats[th_f][img_fold == f]
        results[name] = (stats[best], nested)
        print(f"{name}: best th {best:.3f} PQ {pq(stats[best]):.4f} counts {stats[best].sum(0)[1:].astype(int)}"
              f" | nested PQ {pq(nested):.4f}", flush=True)
    rng = np.random.default_rng(0)
    for k, label in [(0, "best-th"), (1, "nested")]:
        a, b = results["base"][k], results["agree"][k]
        d0 = pq(b) - pq(a)
        ds = []
        for _ in range(args.boot):
            i = rng.integers(0, len(tables), len(tables))
            ds.append(pq(b[i]) - pq(a[i]))
        lo, hi = np.percentile(ds, [2.5, 97.5])
        print(f"{label}: agree - base {d0:+.4f} [{lo:+.4f}, {hi:+.4f}] P>0 {np.mean(np.array(ds) > 0):.2f}")


if __name__ == "__main__":
    main()
