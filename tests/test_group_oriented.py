"""Oriented fragment grouping (PostParams.group_line) against the research implementation."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.dl import PostParams, instances_lowres  # noqa: E402


def _bars() -> np.ndarray:
    prob = np.zeros((128, 128), np.float32)
    prob[60:64, 10:40] = 1.0   # two collinear fragments 14 px apart
    prob[60:64, 54:90] = 1.0
    prob[80:84, 10:40] = 1.0   # a parallel fragment 16 px below the first
    return prob


def test_line_merges_along_axis_not_sideways():
    base = dict(thresh=0.5, min_area=10, mode="hyst", seed_thresh=0.9, group_radius=6)
    assert instances_lowres(_bars(), PostParams(**base)).max() == 3
    inst = instances_lowres(_bars(), PostParams(**base, group_line=16))
    assert inst.max() == 2
    assert inst[62, 20] == inst[62, 70] != inst[82, 20]


def test_matches_research_script():
    import cv2
    from post_research import group_oriented, kept_components

    prob = _bars()
    fg, lab, keep, st = kept_components(prob, 0.5, 0.9, 10)
    ref = group_oriented(fg, lab, keep, st, 16, 6)
    inst = instances_lowres(prob, PostParams(thresh=0.5, min_area=10, mode="hyst", seed_thresh=0.9,
                                             group_radius=6, group_line=16))
    # same partition (labels may be numbered differently)
    pairs = {(a, b) for a, b in zip(ref.ravel(), inst.ravel())}
    assert len(pairs) == len({a for a, _ in pairs}) == len({b for _, b in pairs})
    del cv2
