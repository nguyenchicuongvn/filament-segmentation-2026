"""Panoptic Quality, matching the host's self-evaluation notebook.

Pairs with IoU > 0.5 are true positives (at most one per GT and per prediction,
since IoU > 0.5 matches are unique for non-overlapping predictions). Counts are
pooled across all images before taking the ratio.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pycocotools import mask as mask_util

IOU_THRESHOLD = 0.5


@dataclass
class PQAccumulator:
    tp_iou: list[float] = field(default_factory=list)
    fp: int = 0
    fn: int = 0
    # one-to-many / many-to-one diagnostics: GTs touched by >1 prediction and vice versa
    gt_split: int = 0
    pred_merge: int = 0

    def add(self, gt_rles: list[dict], pred_rles: list[dict]) -> None:
        n_gt, n_pred = len(gt_rles), len(pred_rles)
        if n_gt == 0:
            self.fp += n_pred
            return
        if n_pred == 0:
            self.fn += n_gt
            return
        # pycocotools: iou(dt, gt, iscrowd) -> (n_dt, n_gt)
        iou = np.asarray(mask_util.iou(pred_rles, gt_rles, [0] * n_gt)).T  # (n_gt, n_pred)
        hit = iou > IOU_THRESHOLD
        self.tp_iou.extend(iou[hit].tolist())
        self.fp += int((hit.sum(0) == 0).sum())
        self.fn += int((hit.sum(1) == 0).sum())
        touch = iou > 0
        self.gt_split += int((touch.sum(1) > 1).sum())
        self.pred_merge += int((touch.sum(0) > 1).sum())

    @property
    def tp(self) -> int:
        return len(self.tp_iou)

    @property
    def pq(self) -> float:
        den = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return sum(self.tp_iou) / den if den else 0.0

    @property
    def sq(self) -> float:
        return float(np.mean(self.tp_iou)) if self.tp_iou else 0.0

    def summary(self) -> dict:
        return dict(pq=round(self.pq, 4), sq=round(self.sq, 4), tp=self.tp, fp=self.fp,
                    fn=self.fn, gt_split=self.gt_split, pred_merge=self.pred_merge)
