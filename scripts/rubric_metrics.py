"""The host's quantitative rubric (PLAN card N5) for one pipeline on held-out images.

Overview > Evaluation lists: mean Dice (torchmetrics.segmentation.DiceScore), PQ, the Dice
distribution, the IoU distribution and the 1:n / n:1 distribution. Everything is computed per
annotator entry at full resolution (2048 px), the way the PQ metric scores entries:

- Dice: DiceScore(num_classes=2, include_background=False, input_format='index') on the union of
  the entry's filaments vs the union of the predictions (one value per entry, then the mean);
- PQ / SQ / RQ: src/filseg/metric.py (IoU > 0.5, pooled over entries);
- IoU distribution: for every ground-truth filament, its best IoU with any prediction (0 if none),
  and the IoU of every matched (TP) pair;
- 1:n: for every ground-truth filament, how many predictions overlap it (any shared pixel);
  n:1: for every prediction, how many of the entry's filaments it overlaps. The rates are the
  shares with 2 or more.

Usage:
  python scripts/rubric_metrics.py --probs outputs/colab/unet1024_effb4/probs_val \
      --post configs/post_unet1024_effb4_hyst.json --out docs/logs/rubric_effb4_hyst
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.data import by_image, encode_masks, load_train, with_maps  # noqa: E402
from filseg.dl import PostParams, instances_lowres, load_maps, upsample_labels  # noqa: E402
from filseg.metric import IOU_THRESHOLD, PQAccumulator  # noqa: E402

PROBS: Path
POST: PostParams


def torchmetrics_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Binary Dice of two masks with torchmetrics, as the host computes it (NaN if both are empty)."""
    from torchmetrics.segmentation import DiceScore

    metric = DiceScore(num_classes=2, include_background=False, input_format="index")
    p = torch.from_numpy(pred.astype(np.int64))[None]
    g = torch.from_numpy(gt.astype(np.int64))[None]
    return float(metric(p, g))


def entry_stats(gt_rles: list[dict], pred_rles: list[dict], pred_union: np.ndarray) -> dict:
    """Rubric numbers of one annotator entry."""
    gt_union = (mask_util.decode(gt_rles).max(axis=2) if gt_rles else np.zeros_like(pred_union)).astype(np.uint8)
    if gt_rles and pred_rles:
        iou = np.asarray(mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles))).T   # (n_gt, n_pred)
    else:
        iou = np.zeros((len(gt_rles), len(pred_rles)))
    hit = iou > IOU_THRESHOLD
    touch = iou > 0
    return dict(dice=torchmetrics_dice(pred_union, gt_union),
                best_iou=(iou.max(1) if iou.size else np.zeros(len(gt_rles))).tolist(),
                tp_iou=iou[hit].tolist(),
                one_n=touch.sum(1).tolist(),     # predictions touching each filament
                n_one=touch.sum(0).tolist())     # filaments touched by each prediction


def image_job(args) -> list[tuple[str, list, list, dict]]:
    stem, entries = args
    prob, group = load_maps(PROBS, stem)
    labels = upsample_labels(instances_lowres(prob, POST, group))
    pred = encode_masks(labels)
    union = (labels > 0).astype(np.uint8)
    return [(key, gts, pred, entry_stats(gts, pred, union)) for key, gts in entries]


def init(probs: Path, post: PostParams) -> None:
    global PROBS, POST
    PROBS, POST = probs, post


def summarise(rows: list[dict]) -> dict:
    dice = np.array([r["dice"] for r in rows], float)
    best = np.concatenate([r["best_iou"] for r in rows])
    tp = np.concatenate([r["tp_iou"] for r in rows]) if any(r["tp_iou"] for r in rows) else np.zeros(0)
    one_n = np.concatenate([r["one_n"] for r in rows]).astype(int)
    n_one = np.concatenate([r["n_one"] for r in rows]).astype(int)
    q = [10, 25, 50, 75, 90]
    return {
        "entries": len(rows),
        "dice_mean": float(np.nanmean(dice)),
        "dice_nan_entries": int(np.isnan(dice).sum()),
        "dice_percentiles": dict(zip(map(str, q), np.nanpercentile(dice, q).round(4).tolist())),
        "dice_hist": np.histogram(dice[~np.isnan(dice)], bins=10, range=(0, 1))[0].tolist(),
        "gt_best_iou_hist": np.histogram(best, bins=10, range=(0, 1))[0].tolist(),
        "tp_iou_mean": float(tp.mean()) if len(tp) else 0.0,
        "tp_iou_percentiles": dict(zip(map(str, q), np.percentile(tp, q).round(4).tolist())) if len(tp) else {},
        "one_n_rate": float((one_n >= 2).mean()) if len(one_n) else 0.0,
        "n_one_rate": float((n_one >= 2).mean()) if len(n_one) else 0.0,
        "one_n_counts": {k: int((np.minimum(one_n, 3) == k).sum()) for k in range(4)},
        "n_one_counts": {k: int((np.minimum(n_one, 3) == k).sum()) for k in range(4)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", type=Path, required=True)
    ap.add_argument("--post", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="writes <out>.json and <out>.png")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    post = PostParams(**json.loads(args.post.read_text()))
    groups = [(s, [(t.key, t.gt_rles) for t in es]) for s, es in by_image(with_maps(load_train(), args.probs)).items()]
    with ProcessPoolExecutor(args.jobs, initializer=init, initargs=(args.probs, post)) as ex:
        results = [r for rs in ex.map(image_job, groups) for r in rs]
    acc = PQAccumulator()
    for _, gts, pred, _ in results:
        acc.add(gts, pred)
    summary = {"probs": str(args.probs), "post": post.to_dict(), "images": len(groups),
               "pq": acc.pq, "sq": acc.sq, "rq": acc.pq / acc.sq if acc.sq else 0.0,
               "tp": acc.tp, "fp": acc.fp, "fn": acc.fn, **summarise([r[3] for r in results])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    plot(summary, [r[3] for r in results], args.out.with_suffix(".png"))
    print(json.dumps({k: summary[k] for k in ("images", "entries", "pq", "sq", "rq", "dice_mean", "tp_iou_mean",
                                              "one_n_rate", "n_one_rate")}, indent=1))


def plot(summary: dict, rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    dice = np.array([r["dice"] for r in rows], float)
    ax[0].hist(dice[~np.isnan(dice)], bins=20, range=(0, 1), color="#4c72b0")
    ax[0].set_title(f"Dice per entry (mean {summary['dice_mean']:.3f})")
    ax[1].hist(np.concatenate([r["best_iou"] for r in rows]), bins=20, range=(0, 1), color="#55a868")
    ax[1].axvline(IOU_THRESHOLD, color="k", ls="--", lw=1)
    ax[1].set_title("best IoU per ground-truth filament")
    for a, key, title in ((ax[2], "one_n_counts", "predictions per filament (1:n)"),
                          (ax[3], "n_one_counts", "filaments per prediction (n:1)")):
        c = summary[key]
        a.bar(["0", "1", "2", "3+"], [c[k] for k in range(4)], color="#c44e52")
        a.set_title(title)
    fig.suptitle(f"PQ {summary['pq']:.4f}  SQ {summary['sq']:.3f}  RQ {summary['rq']:.3f}  "
                 f"1:n {summary['one_n_rate']:.3f}  n:1 {summary['n_one_rate']:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
