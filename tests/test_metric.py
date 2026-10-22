import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.data import encode_masks  # noqa: E402
from filseg.metric import PQAccumulator  # noqa: E402


def _rle(mask):
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))


def _box(y0, y1, x0, x1, size=64):
    m = np.zeros((size, size), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_perfect_match():
    gt = [_rle(_box(0, 10, 0, 10)), _rle(_box(20, 30, 20, 30))]
    acc = PQAccumulator()
    acc.add(gt, list(gt))
    assert acc.pq == 1.0 and acc.fp == 0 and acc.fn == 0


def test_fp_fn_and_threshold():
    gt = [_rle(_box(0, 10, 0, 10))]
    # IoU = 50/100 = 0.5 is not > 0.5, so it is both an FP and an FN
    pred = [_rle(_box(0, 10, 0, 5))]
    acc = PQAccumulator()
    acc.add(gt, pred)
    assert (acc.tp, acc.fp, acc.fn) == (0, 1, 1)
    assert acc.pq == 0.0


def test_pq_formula():
    gt = [_rle(_box(0, 10, 0, 10)), _rle(_box(20, 30, 20, 30))]
    pred = [_rle(_box(0, 10, 0, 8)), _rle(_box(40, 50, 40, 50))]  # IoU 0.8 + one FP
    acc = PQAccumulator()
    acc.add(gt, pred)
    assert (acc.tp, acc.fp, acc.fn) == (1, 1, 1)
    assert abs(acc.pq - 0.8 / 2.0) < 1e-6


def test_empty_cases():
    acc = PQAccumulator()
    acc.add([], [_rle(_box(0, 5, 0, 5))])
    acc.add([_rle(_box(0, 5, 0, 5))], [])
    assert (acc.fp, acc.fn) == (1, 1)


def test_encode_masks_roundtrip():
    lab = np.zeros((32, 32), np.int32)
    lab[2:5, 2:5] = 1
    lab[10:20, 10:12] = 2
    rles = encode_masks(lab)
    dec = mask_util.decode(rles)
    assert dec.shape == (32, 32, 2)
    assert dec[..., 0].sum() == 9 and dec[..., 1].sum() == 20
