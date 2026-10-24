"""Train the U-Net filament segmenter.

Usage:
  python scripts/train_unet.py --res 512 --epochs 40 --name unet512
  python scripts/train_unet.py --res 1024 --epochs 60 --bs 4 --split all --name unet1024_full
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.data import encode_masks, load_gray, load_train, split  # noqa: E402
from filseg.dl import (TARGET_CHANNELS, FilamentDataset, PostParams, bce_dice_loss,  # noqa: E402
                       build_model, cached, device, instances, predict_prob)
from filseg.metric import PQAccumulator  # noqa: E402


def validate(model, items, res, dev, post: PostParams) -> PQAccumulator:
    """Quick PQ with fixed cc post-processing; one forward pass per image, scored per annotator."""
    model.eval()
    acc = PQAccumulator()
    preds: dict[str, list] = {}
    for it in items:
        if it.image_id not in preds:
            prob = predict_prob(model, load_gray(it.path), res, dev, tta=False)[0]
            preds[it.image_id] = encode_masks(instances(prob, post))
        acc.add(it.gt_rles, preds[it.image_id])
    model.train()
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder", default="resnet34",
                    help="any segmentation_models_pytorch encoder, e.g. resnet34, efficientnet-b4, tu-convnext_tiny")
    ap.add_argument("--targets", choices=list(TARGET_CHANNELS), default="mask",
                    help="'group' adds a second output channel (see filseg.dl)")
    ap.add_argument("--split", choices=["train", "all"], default="train",
                    help="'all' trains on every image (no held-out val)")
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="unet")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/models"))
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out-dir>/<name>.last.pt (written after every epoch)")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = device()
    data = load_train()
    tr_items = split(data, args.split)
    va_items = split(data, "val") if args.split == "train" else []
    print(f"device={dev} train={len(tr_items)} val={len(va_items)}", flush=True)
    for it in tr_items:  # build cache up front (single process, avoids worker races)
        cached(it, args.res)

    loader = DataLoader(FilamentDataset(tr_items, args.res, augment=True, targets=args.targets),
                        batch_size=args.bs,
                        shuffle=True, num_workers=args.workers, drop_last=True,
                        persistent_workers=args.workers > 0)
    model = build_model(args.encoder, classes=TARGET_CHANNELS[args.targets]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out_dir / f"{args.name}.pt"
    last = args.out_dir / f"{args.name}.last.pt"
    meta = {k: v for k, v in vars(args).items() if k not in ("out_dir", "resume")}
    best, start = -1.0, 1
    if args.resume and last.exists():
        st = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        sched.total_steps = steps  # allows resuming with a larger --epochs
        scaler.load_state_dict(st["scaler"])
        best, start = st["best"], st["epoch"] + 1
        print(f"resumed from {last} at epoch {start} (best val pq {best:.4f})", flush=True)
    post = PostParams()
    for ep in range(start, args.epochs + 1):
        t0, losses = time.time(), []
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = bce_dice_loss(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        msg = f"ep {ep:3d} loss {np.mean(losses):.4f} {time.time() - t0:.0f}s"
        if va_items and (ep % args.val_every == 0 or ep == args.epochs):
            acc = validate(model, va_items, args.res, dev, post)
            msg += f" val {acc.summary()}"
            if acc.pq > best:
                best = acc.pq
                torch.save({"model": model.state_dict(), "meta": meta}, ckpt)
                msg += " *"
        # full training state, written atomically so a session cut mid-write leaves the previous one
        tmp = last.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best}, tmp)
        tmp.replace(last)
        print(msg, flush=True)
    if not va_items:
        torch.save({"model": model.state_dict(), "meta": meta}, ckpt)
    (args.out_dir / f"{args.name}.json").write_text(json.dumps(dict(meta, best_val_pq=best), indent=2))
    print("saved", ckpt, "best val pq", best)


if __name__ == "__main__":
    main()
