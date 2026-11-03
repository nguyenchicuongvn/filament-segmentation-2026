"""Cache Mask R-CNN detections (scores + mask probabilities at the training resolution) for val,
fold or test images, as <out>/<stem>.npz; post-processing and fusion with the U-Net are tuned on
these on CPU (scripts/compare.py specs tune:det / tune:fuse).

Usage:
  python scripts/infer_detections.py --ckpt outputs/models/mrcnn_s1024_f0.pt --which fold --out .../dets_val
  python scripts/infer_detections.py --ckpt outputs/models/mrcnn_s1024_f0.pt --which test --out .../dets_test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.data import fold_split, load_gray, load_train, split, test_paths  # noqa: E402
from filseg.dl import device  # noqa: E402
from filseg.maskrcnn import build_detector, predict_detections, save_detections  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--which", choices=["val", "test", "fold"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N images only (smoke tests)")
    args = ap.parse_args()

    dev = device()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    meta = state["meta"]
    model = build_detector(meta["res"], pretrained=False)
    model.load_state_dict(state["model"])
    model.to(dev).eval()
    if args.which == "fold":
        assert meta.get("fold", -1) >= 0, "--which fold needs a fold checkpoint"
        paths = {t.image_id: t.path for t in fold_split(load_train(), meta["fold"])[1]}
    elif args.which == "val":
        paths = {t.image_id: t.path for t in split(load_train(), "val")}
    else:
        paths = {p.stem: p for p in test_paths()}
    items = sorted(paths.items())[:args.limit or None]
    t0 = time.time()
    for i, (stem, path) in enumerate(items, 1):
        scores, masks = predict_detections(model, load_gray(path), meta["res"], dev)
        save_detections(args.out / f"{stem}.npz", scores, masks)
        if i % 20 == 0 or i == len(items):
            print(f"  {i}/{len(items)} images, {(time.time() - t0) / i:.2f} s/image", flush=True)
    print(f"wrote {len(items)} detection files to {args.out}")


if __name__ == "__main__":
    main()
