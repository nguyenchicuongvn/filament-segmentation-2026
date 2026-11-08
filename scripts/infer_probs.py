"""Save U-Net probability maps (uint8, prediction resolution) for val or test images.

<out>/<stem>.npy is the filament map; <out>/group/<stem>.npy the group map of
--targets group models.

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

from filseg.data import fold_split, load_gray, load_train, split, test_paths  # noqa: E402
from filseg.dl import TARGET_CHANNELS, build_model, device, predict_prob  # noqa: E402


def load_model(ckpts: list[Path], dev):
    models, res, folds = [], None, set()
    for c in ckpts:
        state = torch.load(c, map_location="cpu", weights_only=False)
        meta = state["meta"]
        m = build_model(meta["encoder"], pretrained=False,
                        classes=TARGET_CHANNELS[meta.get("targets", "mask")])
        m.load_state_dict(state["model"])
        models.append(m.to(dev).eval())
        assert res in (None, meta["res"]), "ensemble members must share resolution"
        res = meta["res"]
        folds.add(meta.get("fold", -1))
    return models, res, folds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--which", choices=["val", "test", "fold"], required=True,
                    help="'fold': the held-out images of the checkpoint's fold (its out-of-fold maps)")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--out", type=Path, help="default: outputs/probs/<ckpt stem>_<which>")
    ap.add_argument("--res", type=int, help="infer at this resolution instead of the training one "
                                            "(scale TTA, PLAN 5b 06/11: a 1024 model run at 1152)")
    args = ap.parse_args()

    dev = device()
    models, res, folds = load_model(args.ckpt, dev)
    res = args.res or res
    if args.which == "fold":
        assert len(folds) == 1 and -1 not in folds, f"--which fold needs checkpoints of one fold, got {folds}"
        paths = {t.image_id: t.path for t in fold_split(load_train(), folds.pop())[1]}
    elif args.which == "val":
        paths = {t.image_id: t.path for t in split(load_train(), "val")}
    else:
        paths = {p.stem: p for p in test_paths()}
    out = args.out or Path("outputs/probs") / f"{args.ckpt[0].stem}_{args.which}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for stem, path in sorted(paths.items()):
        gray = load_gray(path)
        maps = [predict_prob(m, gray, res, dev, tta=args.tta) for m in models]
        np.save(out / f"{stem}.npy", np.round(np.mean([x[0] for x in maps], 0) * 255).astype(np.uint8))
        groups = [x[1] for x in maps if len(x) > 1]
        if groups:  # group map, from the members that have one
            (out / "group").mkdir(exist_ok=True)
            np.save(out / "group" / f"{stem}.npy", np.round(np.mean(groups, 0) * 255).astype(np.uint8))
    print(f"{len(paths)} images in {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
