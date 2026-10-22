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
from filseg.dl import (FilamentDataset, PostParams, bce_dice_loss, build_model, cached,  # noqa: E402
                       device, instances, predict_prob)
from filseg.metric import PQAccumulator  # noqa: E402


def validate(model, items, res, dev, post: PostParams) -> PQAccumulator:
    model.eval()
    acc = PQAccumulator()
    for it in items:
        prob = predict_prob(model, load_gray(it.path), res, dev, tta=False)
        acc.add(it.gt_rles, encode_masks(instances(prob, post)))
    model.train()
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--split", choices=["train", "all"], default="train",
                    help="'all' trains on every image (no held-out val)")
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="unet")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/models"))
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = device()
    data = load_train()
    tr_items = split(data, args.split)
    va_items = split(data, "val") if args.split == "train" else []
    print(f"device={dev} train={len(tr_items)} val={len(va_items)}", flush=True)
    for it in tr_items:  # build cache up front (single process, avoids worker races)
        cached(it, args.res)

    loader = DataLoader(FilamentDataset(tr_items, args.res, augment=True), batch_size=args.bs,
                        shuffle=True, num_workers=args.workers, drop_last=True,
                        persistent_workers=args.workers > 0)
    model = build_model(args.encoder).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out_dir / f"{args.name}.pt"
    meta = dict(vars(args), out_dir=str(args.out_dir))
    best = -1.0
    post = PostParams()
    for ep in range(1, args.epochs + 1):
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
        print(msg, flush=True)
    if not va_items:
        torch.save({"model": model.state_dict(), "meta": meta}, ckpt)
    (args.out_dir / f"{args.name}.json").write_text(json.dumps(dict(meta, best_val_pq=best), indent=2))
    print("saved", ckpt, "best val pq", best)


if __name__ == "__main__":
    main()
