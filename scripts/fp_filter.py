"""Train and evaluate the instance false-positive filter on cached val maps.

1. Instances from the val maps with a fixed post-processing config.
2. Label = matches some annotator's filament with IoU > 0.5.
3. 5-fold cross-validation by image -> out-of-fold scores -> the drop threshold
   that maximises val PQ (IoU matrices are computed once, so a threshold sweep is cheap).
4. Fit on all val instances and save (model, threshold) for make_submission.py --filter.

Usage:
  python scripts/fp_filter.py --probs outputs/probs/unet768_r34_val --post configs/post_unet768.json \
      --out configs/fpfilter_unet768.pkl
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
from pycocotools import mask as mask_util
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.data import encode_masks, load_gray, load_train, split  # noqa: E402
from filseg.dl import PostParams, instances_lowres, load_maps, upsample_labels  # noqa: E402
from filseg.instfilter import FEATURES, instance_features  # noqa: E402
from filseg.metric import IOU_THRESHOLD  # noqa: E402


def _image_table(args):
    """Features and per-annotator IoU matrices for one val image."""
    probs_dir, post, path, stem, gts = args
    prob, group = load_maps(Path(probs_dir), stem)
    inst = instances_lowres(prob, post, group)
    feats = instance_features(inst, prob, load_gray(path), group)
    rles = encode_masks(upsample_labels(inst))
    ious = []
    for gt in gts:  # one (n_gt, n_pred) matrix per annotator entry
        if gt and rles:
            ious.append(np.asarray(mask_util.iou(rles, gt, [0] * len(gt))).T)
        else:
            ious.append(np.zeros((len(gt), len(rles))))
    label = np.zeros(len(rles), bool)
    for m in ious:
        if m.size:
            label |= (m > IOU_THRESHOLD).any(0)
    return stem, feats, label, ious


def pq_with_keep(tables, keep_by_stem) -> tuple[float, dict]:
    tp_iou, fp, fn = [], 0, 0
    for stem, _, _, ious in tables:
        keep = keep_by_stem[stem]
        for m in ious:
            m = m[:, keep]
            hit = m > IOU_THRESHOLD
            tp_iou.extend(m[hit].tolist())
            fp += int((hit.sum(0) == 0).sum())
            fn += int((hit.sum(1) == 0).sum())
    den = len(tp_iou) + 0.5 * fp + 0.5 * fn
    return (sum(tp_iou) / den if den else 0.0), dict(tp=len(tp_iou), fp=fp, fn=fn)


def make_clf():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", type=Path, required=True, help="cached val maps")
    ap.add_argument("--post", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    post = PostParams(**json.loads(args.post.read_text()))
    by_img: dict[str, tuple[Path, list]] = {}
    for it in split(load_train(), "val"):
        by_img.setdefault(it.image_id, (it.path, []))[1].append(it.gt_rles)
    jobs = [(str(args.probs), post, p, s, gts) for s, (p, gts) in sorted(by_img.items())]
    with ProcessPoolExecutor() as pool:
        tables = list(pool.map(_image_table, jobs, chunksize=4))

    stems = [t[0] for t in tables]
    X = np.concatenate([t[1] for t in tables])
    y = np.concatenate([t[2] for t in tables])
    owner = np.repeat(np.arange(len(tables)), [len(t[2]) for t in tables])
    print(f"{len(stems)} images, {len(y)} instances, {y.mean():.2f} positive", flush=True)

    # out-of-fold scores, folds by image
    idx = list(range(len(stems)))
    random.Random(0).shuffle(idx)
    fold_of = np.empty(len(stems), int)
    for f, chunk in enumerate(np.array_split(idx, args.folds)):
        fold_of[chunk] = f
    oof = np.zeros(len(y))
    for f in range(args.folds):
        te = fold_of[owner] == f
        oof[te] = make_clf().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]

    def keep_at(th):
        out, start = {}, 0
        for t in tables:
            n = len(t[2])
            out[t[0]] = oof[start:start + n] >= th
            start += n
        return out

    base, base_c = pq_with_keep(tables, keep_at(-1.0))
    print(f"no filter: pq {base:.4f} {base_c}")
    best = (base, -1.0, base_c)
    for th in np.arange(0.05, 0.75, 0.025):
        pq, c = pq_with_keep(tables, keep_at(th))
        print(f"th {th:.3f}: pq {pq:.4f} {c}")
        if pq > best[0]:
            best = (pq, float(th), c)
    print(f"best: pq {best[0]:.4f} at th {best[1]:.3f} {best[2]} (no filter {base:.4f})")

    if args.out:
        clf = make_clf().fit(X, y)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "wb") as fh:
            pickle.dump({"model": clf, "threshold": best[1], "features": FEATURES,
                         "post": post.to_dict(), "cv_pq": best[0], "base_pq": base}, fh)
        print("->", args.out)


if __name__ == "__main__":
    main()
