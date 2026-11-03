"""Rubric metrics (docs/PLAN.md N5): torchmetrics Dice and the per-entry overlap statistics."""
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rubric_metrics import entry_stats, summarise, torchmetrics_dice  # noqa: E402


def _rle(m):
    return mask_util.encode(np.asfortranarray(m.astype(np.uint8)))


def test_torchmetrics_dice_matches_formula():
    rng = np.random.default_rng(0)
    for _ in range(5):
        a, b = rng.random((64, 64)) > 0.6, rng.random((64, 64)) > 0.7
        assert abs(torchmetrics_dice(a, b) - 2 * (a & b).sum() / (a.sum() + b.sum())) < 1e-6
    z = np.zeros((8, 8), bool)
    assert np.isnan(torchmetrics_dice(z, z)) and torchmetrics_dice(z, ~z) == 0


def test_entry_stats_overlaps():
    g1, g2 = np.zeros((32, 32), bool), np.zeros((32, 32), bool)
    g1[2:6, 2:20], g2[20:24, 2:20] = True, True
    p1 = np.zeros((32, 32), bool); p1[2:6, 2:10] = True      # half of g1
    p2 = np.zeros((32, 32), bool); p2[2:6, 10:20] = True     # other half of g1 (g1 is split: 1:n = 2)
    p3 = np.zeros((32, 32), bool); p3[20:24, 2:20] = True    # g2 exactly
    preds = [_rle(p1), _rle(p2), _rle(p3)]
    s = entry_stats([_rle(g1), _rle(g2)], preds, (p1 | p2 | p3).astype(np.uint8))
    assert s["one_n"] == [2, 1] and s["n_one"] == [1, 1, 1]
    # p2 covers 40 of g1's 72 pixels: IoU 0.556 > 0.5, a TP despite the split
    assert np.allclose(sorted(s["tp_iou"]), [40 / 72, 1.0]) and np.allclose(s["best_iou"], [40 / 72, 1.0])
    assert abs(s["dice"] - 1.0) < 1e-6   # unions are identical
    out = summarise([s])
    assert out["one_n_rate"] == 0.5 and out["n_one_rate"] == 0.0
    assert abs(out["tp_iou_mean"] - (40 / 72 + 1) / 2) < 1e-6
