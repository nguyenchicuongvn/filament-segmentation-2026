"""Turn cached test probability maps into a submission CSV.

Usage:
  python scripts/make_submission.py --probs outputs/probs/unet512_r34_test \
      --post configs/post_unet512.json --out outputs/submission_unet512.csv
  python scripts/make_submission.py --probs outputs/probs/unet768_r34_test \
      --filter configs/fpfilter_unet768.pkl --out outputs/submission_unet768_filt.csv
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.data import encode_masks, load_gray, rle_counts_str, test_paths  # noqa: E402
from filseg.dl import PostParams, instances_lowres, load_maps, upsample_labels  # noqa: E402
from filseg.instfilter import instance_features  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", type=Path, required=True)
    ap.add_argument("--post", type=Path)
    ap.add_argument("--filter", type=Path, help="FP filter from scripts/fp_filter.py (uses its own post config)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    filt = pickle.loads(args.filter.read_bytes()) if args.filter else None
    if filt:
        p = PostParams(**filt["post"])
    else:
        p = PostParams(**json.loads(args.post.read_text())) if args.post else PostParams()
    rows = []
    for path in test_paths():
        prob, group = load_maps(args.probs, path.stem)
        inst = instances_lowres(prob, p, group)
        if filt and inst.max() > 0:
            score = filt["model"].predict_proba(instance_features(inst, prob, load_gray(path), group))[:, 1]
            keep = np.concatenate([[False], score >= filt["threshold"]])
            lut = np.zeros(len(keep), np.int32)
            lut[keep] = np.arange(1, int(keep.sum()) + 1)
            inst = lut[inst]
        for i, rle in enumerate(encode_masks(upsample_labels(inst))):
            rows.append((f"{path.stem}_{i + 1}", rle_counts_str(rle)))
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{df.filament_id.str.split('_').str[0].nunique()} images, {len(df)} filaments -> {args.out}")


if __name__ == "__main__":
    main()
