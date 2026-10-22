"""Save U-Net probability maps (uint8, prediction resolution) for val or test images.

Post-processing is tuned on these cached maps, so the network runs only once.

Usage:
  python scripts/infer_probs.py --ckpt outputs/models/unet512_r34.pt --which val
  python scripts/infer_probs.py --ckpt outputs/models/unet512_r34.pt --which test --tta
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.data import load_gray, load_train, split, test_paths  # noqa: E402
from filseg.dl import build_model, device, predict_prob  # noqa: E402


def load_model(ckpts: list[Path], dev):
    models, res = [], None
    for c in ckpts:
        state = torch.load(c, map_location="cpu", weights_only=False)
        meta = state["meta"]
        m = build_model(meta["encoder"], pretrained=False)
        m.load_state_dict(state["model"])
        models.append(m.to(dev).eval())
        assert res in (None, meta["res"]), "ensemble members must share resolution"
        res = meta["res"]
    return models, res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--which", choices=["val", "test"], required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--out", type=Path, help="default: outputs/probs/<ckpt stem>_<which>")
    args = ap.parse_args()

    dev = device()
    models, res = load_model(args.ckpt, dev)
    if args.which == "val":
        paths = {t.image_id: t.path for t in split(load_train(), "val")}
    else:
        paths = {p.stem: p for p in test_paths()}
    out = args.out or Path("outputs/probs") / f"{args.ckpt[0].stem}_{args.which}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for stem, path in sorted(paths.items()):
        gray = load_gray(path)
        prob = np.mean([predict_prob(m, gray, res, dev, tta=args.tta) for m in models], axis=0)
        np.save(out / f"{stem}.npy", np.round(prob * 255).astype(np.uint8))
    print(f"{len(paths)} images in {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
