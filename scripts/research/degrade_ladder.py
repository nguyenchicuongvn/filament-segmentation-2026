"""Calibrated degradation ladder (PLAN 5b, 06/11): drop the q lowest-scoring instances of a pipeline.

Instances = make_submission.py's pipeline (post config of the FP filter, then the filter's keep
test); each kept instance's filter score is its confidence. The q-quantile of the scores over all
kept instances of the set is the cut, so the same q removes about the same share on OOF and test.

  --oof DIR    OOF maps: prints PQ (pooled over annotator entries) for every q, and a paired
               bootstrap by image of each q against q = 0
  --test DIR   test maps: writes outputs/submission_<tag>_drop<q>.csv for every q > 0

The LB readings of the three CSVs against their OOF deltas give the test-side slope (an LB that
cannot resolve 0.003 should still resolve 0.02-0.10).

Usage:
  python scripts/research/degrade_ladder.py --filter configs/fpfilter_ens_n1n2.pkl \
      --oof outputs/ens/n1n2_oof --test outputs/ens/n1n2_test --tag ens_n1n2
"""
from __future__ import annotations

import argparse
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from filseg.data import by_image, encode_masks, load_gray, load_train, rle_counts_str, test_paths, with_maps  # noqa: E402
from filseg.dl import PostParams, instances_lowres, load_maps, upsample_labels  # noqa: E402
from filseg.instfilter import instance_features, score  # noqa: E402
from filseg.metric import IOU_THRESHOLD  # noqa: E402

QS = (0.0, 0.15, 0.30, 0.50)


def _kept(args) -> tuple[str, list[dict], np.ndarray]:
    probs_dir, stem, path, filt = args
    prob, group = load_maps(Path(probs_dir), stem)
    inst = instances_lowres(prob, PostParams(**filt["post"]), group)
    if inst.max() == 0:
        return stem, [], np.zeros(0)
    sc = score(filt, instance_features(inst, prob, load_gray(Path(path)), group))
    keep = sc >= filt["threshold"]
    lut = np.zeros(len(sc) + 1, np.int32)
    lut[1:][keep] = np.arange(1, int(keep.sum()) + 1)
    return stem, encode_masks(upsample_labels(lut[inst])), sc[keep]


def _run(probs_dir: Path, stems_paths: list[tuple[str, Path]], filt: dict, jobs: int):
    with ProcessPoolExecutor(jobs) as ex:
        return {s: (r, sc) for s, r, sc in ex.map(_kept, [(str(probs_dir), s, str(p), filt) for s, p in stems_paths])}


def _image_stats(gts: list[list[dict]], rles: list[dict]) -> tuple[float, int, int, int]:
    """Sum of TP IoUs, TP, FP, FN of one image pooled over its annotator entries."""
    s = tp = fp = fn = 0
    for gt in gts:
        if not gt:
            fp += len(rles)
            continue
        if not rles:
            fn += len(gt)
            continue
        iou = np.asarray(mask_util.iou(rles, gt, [0] * len(gt))).T
        hit = iou > IOU_THRESHOLD
        s += float(iou[hit].sum()); tp += int(hit.sum())
        fp += int((hit.sum(0) == 0).sum()); fn += int((hit.sum(1) == 0).sum())
    return s, tp, fp, fn


def _pq(a: np.ndarray) -> float:
    s, tp, fp, fn = a.sum(0)
    return s / (tp + 0.5 * fp + 0.5 * fn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", type=Path, required=True)
    ap.add_argument("--oof", type=Path)
    ap.add_argument("--test", type=Path)
    ap.add_argument("--tag", default="ladder")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    filt = pickle.loads(args.filter.read_bytes())

    if args.oof:
        imgs = by_image(with_maps(load_train(), args.oof))
        res = _run(args.oof, [(s, es[0].path) for s, es in imgs.items()], filt, args.jobs)
        allsc = np.concatenate([sc for _, sc in res.values()])
        stems = sorted(imgs)
        tab = {}
        for q in QS:
            cut = np.quantile(allsc, q) if q else -np.inf
            tab[q] = np.array([_image_stats([e.gt_rles for e in imgs[s]],
                                            [r for r, c in zip(*res[s]) if c >= cut]) for s in stems])
        rng = np.random.default_rng(0)
        boot = [rng.integers(0, len(stems), len(stems)) for _ in range(2000)]
        print(f"{len(stems)} images, {len(allsc)} kept instances, filter {args.filter.name}")
        for q in QS:
            d = np.array([_pq(tab[q][b]) - _pq(tab[0.0][b]) for b in boot])
            s, tp, fp, fn = tab[q].sum(0)
            print(f"  drop {q:.2f}  PQ {_pq(tab[q]):.4f}  TP {tp} FP {fp} FN {fn}  "
                  f"d {_pq(tab[q]) - _pq(tab[0.0]):+.4f} [{np.quantile(d, .025):+.4f}, {np.quantile(d, .975):+.4f}]")

    if args.test:
        res = _run(args.test, [(p.stem, p) for p in test_paths()], filt, args.jobs)
        allsc = np.concatenate([sc for _, sc in res.values()])
        for q in QS[1:]:
            cut = np.quantile(allsc, q)
            rows = []
            for stem in sorted(res):
                kept = [r for r, c in zip(*res[stem]) if c >= cut]
                rows += [(f"{stem}_{i + 1}", rle_counts_str(r)) for i, r in enumerate(kept)]
            out = Path(f"outputs/submission_{args.tag}_drop{int(q * 100)}.csv")
            pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(out, index=False)
            print(f"  test drop {q:.2f}: {len(rows)} filaments -> {out}")


if __name__ == "__main__":
    main()
