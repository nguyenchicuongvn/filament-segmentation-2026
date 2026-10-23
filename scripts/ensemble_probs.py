"""Average cached probability maps from several models (resized to the largest resolution).

Usage:
  python scripts/ensemble_probs.py --inputs outputs/probs/unet512_r34_val outputs/probs/unet768_r34_val \
      --weights 1 1 --out outputs/probs/ens_512_768_val
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    weights = np.array(args.weights or [1.0] * len(args.inputs), np.float32)
    assert len(weights) == len(args.inputs)
    weights /= weights.sum()
    stems = sorted(p.stem for p in args.inputs[0].glob("*.npy"))
    args.out.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        maps = [np.load(d / f"{stem}.npy").astype(np.float32) for d in args.inputs]
        res = max(m.shape[0] for m in maps)
        acc = sum(w * (m if m.shape[0] == res else cv2.resize(m, (res, res), interpolation=cv2.INTER_LINEAR))
                  for w, m in zip(weights, maps))
        np.save(args.out / f"{stem}.npy", np.round(acc).astype(np.uint8))
    print(f"{len(stems)} maps -> {args.out}")


if __name__ == "__main__":
    main()
