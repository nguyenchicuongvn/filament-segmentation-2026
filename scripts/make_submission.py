"""Turn cached test probability maps into a submission CSV.

Usage:
  python scripts/make_submission.py --probs outputs/probs/unet512_r34_test \
      --post configs/post_unet512.json --out outputs/submission_unet512.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.data import encode_masks, rle_counts_str, test_paths  # noqa: E402
from filseg.dl import PostParams, instances  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", type=Path, required=True)
    ap.add_argument("--post", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    p = PostParams(**json.loads(args.post.read_text())) if args.post else PostParams()
    rows = []
    for path in test_paths():
        prob = np.load(args.probs / f"{path.stem}.npy").astype(np.float32) / 255.0
        for i, rle in enumerate(encode_masks(instances(prob, p))):
            rows.append((f"{path.stem}_{i + 1}", rle_counts_str(rle)))
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{df.filament_id.str.split('_').str[0].nunique()} images, {len(df)} filaments -> {args.out}")


if __name__ == "__main__":
    main()
