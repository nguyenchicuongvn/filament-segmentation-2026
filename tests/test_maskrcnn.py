"""Mask R-CNN member (docs/PLAN.md N4): targets, detections -> instances, fusion with the U-Net."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.maskrcnn import (DetParams, FuseParams, build_detector, detections_to_labels, fuse_with_unet,
                             targets_from_labels)


def _box(h, w, y0, y1, x0, x1, val=255):
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = val
    return m


def test_targets_from_labels_boxes_and_masks():
    lab = np.zeros((32, 32), np.int16)
    lab[2:5, 3:10] = 1
    lab[20:30, 20:22] = 4
    lab[0, 0] = 7  # below min_area: dropped
    t = targets_from_labels(lab)
    assert t["boxes"].tolist() == [[3, 2, 10, 5], [20, 20, 22, 30]]
    assert t["masks"].shape == (2, 32, 32) and t["masks"][1].sum() == 20
    assert t["labels"].tolist() == [1, 1]
    empty = targets_from_labels(np.zeros((8, 8), np.int16))
    assert empty["boxes"].shape == (0, 4) and empty["masks"].shape == (0, 8, 8)


def test_detections_to_labels_score_order_and_hidden():
    a = _box(64, 64, 0, 20, 0, 20)
    b = _box(64, 64, 5, 20, 5, 20)       # entirely inside a: hidden, dropped
    c = _box(64, 64, 40, 60, 40, 60)
    lab = detections_to_labels(np.array([0.9, 0.8, 0.3]), np.stack([a, b, c]), DetParams(min_area=10))
    assert lab.max() == 1 and (lab == 1).sum() == 400   # c is below the score threshold
    lab = detections_to_labels(np.array([0.9, 0.8, 0.6]), np.stack([a, b, c]), DetParams(min_area=10))
    assert lab.max() == 2
    # disjoint by construction
    assert detections_to_labels(np.zeros(0), np.zeros((0, 64, 64), np.uint8), DetParams()).max() == 0


def test_fuse_splits_a_merged_component():
    inst = np.zeros((64, 64), np.int32)
    inst[10:14, 0:60] = 1                     # one U-Net component spanning two filaments
    d1 = _box(64, 64, 9, 15, 0, 28)
    d2 = _box(64, 64, 9, 15, 32, 60)
    out = fuse_with_unet(inst, np.array([0.9, 0.9]), np.stack([d1, d2]), FuseParams(min_area=5))
    assert out.max() == 2
    assert len(np.unique(out[10:14, 0:20])) == 1 and len(np.unique(out[10:14, 40:60])) == 1
    assert out[11, 5] != out[11, 50]
    assert ((out > 0) == (inst > 0)).all()    # pixels unchanged, only the grouping


def test_fuse_merges_fragments_and_respects_flags():
    inst = np.zeros((64, 64), np.int32)
    inst[10:14, 0:20] = 1
    inst[10:14, 24:44] = 2                    # two fragments of one filament
    inst[50:54, 50:60] = 3                    # a component no detection touches
    d = _box(64, 64, 8, 16, 0, 44)
    out = fuse_with_unet(inst, np.array([0.95]), d[None], FuseParams(min_area=5))
    assert out[11, 5] == out[11, 30] and out[51, 55] not in (0, out[11, 5])
    out = fuse_with_unet(inst, np.array([0.95]), d[None], FuseParams(min_area=5, merge=False))
    assert out[11, 5] != out[11, 30]
    out = fuse_with_unet(inst, np.array([0.95]), d[None], FuseParams(min_area=5, require_det=True))
    assert out[51, 55] == 0 and out[11, 5] > 0
    # a confident detection the U-Net missed is added only when add_thresh allows it
    miss = _box(64, 64, 30, 36, 30, 50)
    two = np.stack([d, miss])
    assert fuse_with_unet(inst, np.array([0.95, 0.95]), two, FuseParams(min_area=5))[32, 40] == 0
    assert fuse_with_unet(inst, np.array([0.95, 0.95]), two, FuseParams(min_area=5, add_thresh=0.9))[32, 40] > 0


def test_detector_trains_and_predicts_on_cpu():
    model = build_detector(128, pretrained=False)
    lab = np.zeros((128, 128), np.int16)
    lab[20:30, 10:90] = 1
    x = torch.rand(3, 128, 128)
    model.train()
    loss = sum(model([x], [targets_from_labels(lab)]).values())
    loss.backward()
    assert torch.isfinite(loss)
    model.eval()
    with torch.no_grad():
        out = model([x])[0]
    assert set(out) >= {"boxes", "scores", "masks"}
