import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from filseg.instfilter import AGREE_FEATURES, FEATURES, agreement_features, select  # noqa: E402

COL = {n: i for i, n in enumerate(AGREE_FEATURES)}


def _two_instances():
    """64x64 labels: instance 1 = rows 10-19 x cols 10-19 (100 px), instance 2 = rows 40-49 x cols 30-39."""
    inst = np.zeros((64, 64), np.int32)
    inst[10:20, 10:20] = 1
    inst[40:50, 30:40] = 2
    return inst


def test_hand_computed():
    inst = _two_instances()
    base = np.where(inst > 0, 0.9, 0.0).astype(np.float32)
    # m1: covers instance 1 exactly (p 0.8), nothing on instance 2
    m1 = np.zeros((64, 64), np.float32)
    m1[10:20, 10:20] = 0.8
    # m2: left half of instance 1 (50 px at 0.6), the rest 0.2; instance 2 at 0.7 plus 10 px outside it
    m2 = np.zeros((64, 64), np.float32)
    m2[10:20, 10:20] = 0.2
    m2[10:20, 10:15] = 0.6
    m2[40:50, 30:41] = 0.7
    # m3: all zero; m4: 0.55 on both instances but instance 2 only as a 5x5 (25 px < 50: dropped as a component)
    m3 = np.zeros((64, 64), np.float32)
    m4 = np.zeros((64, 64), np.float32)
    m4[10:20, 10:20] = 0.55
    m4[40:45, 30:35] = 0.55
    f = agreement_features(inst, base, [m1, m2, m3, m4])
    assert f.shape == (2, len(AGREE_FEATURES))

    # instance 1
    means1 = [0.9, 0.8, (50 * 0.6 + 50 * 0.2) / 100, 0.0, 0.55]
    assert f[0, :4] == pytest.approx(means1[1:], abs=1e-6)
    assert f[0, COL["agree_std5"]] == pytest.approx(np.std(means1), abs=1e-6)
    assert f[0, COL["agree_min4"]] == pytest.approx(0.0, abs=1e-6)
    # votes > 0.5: base, m1, m4 everywhere (3) + m2 on the left half -> >= 3 on every pixel
    assert f[0, COL["agree_vote3"]] == pytest.approx(1.0)
    # best IoU: m1 1.0; m2 component = left half 50 px -> 50/100; m3 0; m4 1.0
    assert f[0, COL["agree_iou4"]] == pytest.approx((1.0 + 0.5 + 0.0 + 1.0) / 4)

    # instance 2
    means2 = [0.9, 0.0, 0.7, 0.0, 25 * 0.55 / 100]
    assert f[1, :4] == pytest.approx(means2[1:], abs=1e-6)
    assert f[1, COL["agree_std5"]] == pytest.approx(np.std(means2), abs=1e-6)
    assert f[1, COL["agree_min4"]] == pytest.approx(0.0, abs=1e-6)
    # votes: base + m2 everywhere (2), + m4 on the 5x5 corner (3) -> 25 / 100
    assert f[1, COL["agree_vote3"]] == pytest.approx(0.25)
    # m2 component is 10x11 = 110 px containing the 100 px instance -> 100/110; m4's is too small
    assert f[1, COL["agree_iou4"]] == pytest.approx((0.0 + 100 / 110 + 0.0 + 0.0) / 4)


def test_identical_models():
    inst = _two_instances()
    rng = np.random.default_rng(0)
    base = np.where(inst > 0, rng.uniform(0.6, 1.0, inst.shape), rng.uniform(0.0, 0.4, inst.shape))
    base = base.astype(np.float32)
    f = agreement_features(inst, base, [base] * 4)
    assert f[:, COL["agree_std5"]] == pytest.approx(0.0, abs=1e-6)
    assert f[:, COL["agree_iou4"]] == pytest.approx(1.0)
    assert f[:, COL["agree_vote3"]] == pytest.approx(1.0)


def test_empty_and_wrong_count():
    inst = np.zeros((64, 64), np.int32)
    p = np.zeros((64, 64), np.float32)
    assert agreement_features(inst, p, [p] * 4).shape == (0, len(AGREE_FEATURES))
    with pytest.raises(ValueError):
        agreement_features(_two_instances(), p, [p] * 3)


def test_select_keeps_old_filters():
    feats = np.arange(2 * (len(FEATURES) + len(AGREE_FEATURES)), dtype=np.float32).reshape(2, -1)
    assert np.array_equal(select(feats, FEATURES), feats[:, :len(FEATURES)])
    assert np.array_equal(select(feats[:, :len(FEATURES)], FEATURES[:14]), feats[:, :14])
    assert np.array_equal(select(feats, ["agree_iou4"]), feats[:, -1:])


def test_crosstune_unchanged_without_filter():
    """evaluate_crosstuned after the refactor equals the original loop on synthetic per-combo stats."""
    import compare

    rng = np.random.default_rng(1)
    n, n_combo = 31, 5
    per_combo = rng.integers(0, 20, size=(n_combo, n, 4)).astype(float)
    stems = [f"s{i}" for i in range(n)]
    compare._GRID_MEMO.clear()
    orig = compare._grid_stats
    compare._grid_stats = lambda *a: ([compare.PostParams() for _ in range(n_combo)], per_combo)
    try:
        got = compare.evaluate_crosstuned("x", "hyst", stems, {})
    finally:
        compare._grid_stats = orig
    want = np.zeros((n, 4))
    for r in range(3):  # the pre-refactor loop
        half = np.random.default_rng(100 + r).permutation(n) < n // 2
        for tune_on in (half, ~half):
            best = int(np.argmax([compare.pq(per_combo[c][tune_on].sum(0)) for c in range(n_combo)]))
            want[~tune_on] += per_combo[best][~tune_on]
    assert np.array_equal(got, want / 3)
