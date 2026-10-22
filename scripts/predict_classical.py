"""Run the classical baseline on the test images and write a submission CSV.

Usage:
  python scripts/predict_classical.py --params outputs/best_params.json --out outputs/submission.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.classical import Params, segment  # noqa: E402
from filseg.data import encode_masks, load_gray, rle_counts_str, test_paths  # noqa: E402


def _predict(path_and_params):
    path, p = path_and_params
    rles = encode_masks(segment(load_gray(path), p))
    return [(f"{path.stem}_{i + 1}", rle_counts_str(r)) for i, r in enumerate(rles)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=Path)
    ap.add_argument("--out", type=Path, default=Path("outputs/submission.csv"))
    args = ap.parse_args()

    p = Params(**json.loads(args.params.read_text())) if args.params else Params()
    paths = test_paths()
    t0 = time.time()
    with ProcessPoolExecutor() as pool:
        rows = [row for rows in pool.map(_predict, [(x, p) for x in paths]) for row in rows]
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(paths)} images -> {len(df)} filaments in {time.time() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
