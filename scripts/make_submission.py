"""Turn cached test probability maps into a submission CSV.

Usage:
  python scripts/make_submission.py --probs outputs/probs/unet512_r34_test \
      --post configs/post_unet512.json --out outputs/submission_unet512.csv
  python scripts/make_submission.py --probs outputs/probs/unet768_r34_test \
      --filter configs/fpfilter_unet768.pkl --out outputs/submission_unet768_filt.csv
A filter trained with agreement features (fp_filter.py --agree-maps) needs --agree-maps with the
same models' test maps, in the same order.
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
from filseg.instfilter import AGREE_FEATURES, agreement_features, instance_features, score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", type=Path, required=True)
    ap.add_argument("--post", type=Path)
    ap.add_argument("--filter", type=Path, help="FP filter from scripts/fp_filter.py (uses its own post config)")
    ap.add_argument("--agree-maps", help="comma-separated test maps of the 4 other models (agreement filter)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    filt = pickle.loads(args.filter.read_bytes()) if args.filter else None
    if filt:
        p = PostParams(**filt["post"])
    else:
        p = PostParams(**json.loads(args.post.read_text())) if args.post else PostParams()
    agree = args.agree_maps.split(",") if args.agree_maps else None
    if filt and any(n in AGREE_FEATURES for n in filt["features"]) and not agree:
        raise SystemExit("this filter uses agreement features: pass --agree-maps")
    rows = []
    for path in test_paths():
        prob, group = load_maps(args.probs, path.stem)
        inst = instances_lowres(prob, p, group)
        if filt and inst.max() > 0:
            feats = instance_features(inst, prob, load_gray(path), group)
            if agree:
                others = [load_maps(Path(d), path.stem)[0] for d in agree]
                feats = np.concatenate([feats, agreement_features(inst, prob, others)], 1)
            sc = score(filt, feats)
            keep = np.concatenate([[False], sc >= filt["threshold"]])
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
