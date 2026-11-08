"""Train the Mask R-CNN instance member (PLAN card N4).

Same conventions as train_unet.py, so the Colab notebook and the watcher treat it like a U-Net run:
--fold / --split, --resume from <name>.last.pt, best checkpoint by val PQ in <name>.pt, a
<name>.history.csv with the same leading columns, and '@@ ' JSON progress lines. Validation scores
the detector alone (detections_to_labels at score 0.5) on the held-out images, without TTA.

Usage:
  python scripts/train_maskrcnn.py --res 1024 --epochs 24 --fold 0 --name mrcnn_s1024_f0
  python scripts/train_maskrcnn.py --res 512 --split train --epochs 12 --name mrcnn_512_holdout  # Mac
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.data import encode_masks, fold_split, load_gray, load_train, split  # noqa: E402
from filseg.dl import device, upsample_labels  # noqa: E402
from filseg.maskrcnn import (DetectionDataset, DetParams, build_detector, collate,  # noqa: E402
                             detections_to_labels, instance_cached, predict_detections)
from filseg.metric import PQAccumulator  # noqa: E402

HISTORY_FIELDS = ["epoch", "loss", "lr", "epoch_s", "val_s", "elapsed_s", "val_pq", "val_sq", "val_dice",
                  "tp", "fp", "fn", "gt_split", "pred_merge"]


def fmt_dur(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def validate(model, items, res, dev) -> PQAccumulator:
    acc, cache = PQAccumulator(), {}
    for it in items:
        if it.image_id not in cache:
            scores, masks = predict_detections(model, load_gray(it.path), res, dev)
            cache[it.image_id] = encode_masks(upsample_labels(detections_to_labels(scores, masks, DetParams(
                min_area=max(1, round(50 * (res / 1024) ** 2))))))
        acc.add(it.gt_rles, cache[it.image_id])
    model.train()
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--split", choices=["train", "all"], default="train")
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--val-every", type=int, default=4)
    ap.add_argument("--max-train", type=int, default=0, help="use only the first N train entries (smoke tests)")
    ap.add_argument("--max-val", type=int, default=0, help="score only the first N val entries (smoke tests)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="mrcnn")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/models"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    # accepted for symmetry with train_unet.py command lines (the notebook passes them); unused here
    ap.add_argument("--encoder", default="maskrcnn_r50_fpn_v2")
    ap.add_argument("--targets", default="instances")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    dev = device()
    data = load_train()
    if args.fold >= 0:
        tr_items, va_items = fold_split(data, args.fold)
    else:
        tr_items = split(data, args.split)
        va_items = split(data, "val") if args.split == "train" else []
    if args.max_train:
        tr_items = tr_items[:args.max_train]
    if args.max_val:
        va_items = va_items[:args.max_val]
    print(f"device={dev} train={len(tr_items)} val={len(va_items)}", flush=True)
    for it in tr_items:  # build the cache up front (single process, avoids worker races)
        instance_cached(it, args.res)
    loader = DataLoader(DetectionDataset(tr_items, args.res, augment=True), batch_size=args.bs, shuffle=True,
                        num_workers=args.workers, drop_last=True, collate_fn=collate,
                        persistent_workers=args.workers > 0)
    model = build_detector(args.res, pretrained=not args.no_pretrained).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out_dir / f"{args.name}.pt"
    last = args.out_dir / f"{args.name}.last.pt"
    meta = {k: v for k, v in vars(args).items() if k not in ("out_dir", "resume")}
    meta["kind"] = "maskrcnn"
    hist_path = args.out_dir / f"{args.name}.history.csv"
    best, start, elapsed0 = -1.0, 1, 0.0
    if args.resume and last.exists():
        st = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        sched.total_steps = steps
        scaler.load_state_dict(st["scaler"])
        best, start, elapsed0 = st["best"], st["epoch"] + 1, st.get("elapsed", 0.0)
        print(f"resumed from {last} at epoch {start} (best val pq {best:.4f})", flush=True)
    if start == 1 or not hist_path.exists():
        with open(hist_path, "w", newline="") as fh:
            csv.writer(fh).writerow(HISTORY_FIELDS)
    n_steps = len(loader)
    report_every = max(1, n_steps // 4)
    t_start, epoch_times = time.time(), []
    print(f"{args.name}: epochs {start}..{args.epochs}, {n_steps} steps/epoch, batch {args.bs}, res {args.res}, "
          f"Mask R-CNN R50-FPN v2", flush=True)
    model.train()
    for ep in range(start, args.epochs + 1):
        t0, losses = time.time(), []
        for step, (xs, ts) in enumerate(loader, 1):
            xs = [x.to(dev, non_blocking=True) for x in xs]
            ts = [{k: v.to(dev, non_blocking=True) for k, v in t.items()} for t in ts]
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                parts = model(xs, ts)
                loss = sum(parts.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss.item()))
            if step % report_every == 0 and step < n_steps:
                dt = time.time() - t0
                print(f"  ep {ep}/{args.epochs} step {step}/{n_steps} loss {np.mean(losses):.4f} "
                      f"{step / dt:.2f} it/s, epoch done in ~{fmt_dur(dt / step * (n_steps - step))}", flush=True)
        epoch_s = time.time() - t0
        epoch_times.append(epoch_s)
        row = dict(epoch=ep, loss=round(float(np.mean(losses)), 5), lr=f"{sched.get_last_lr()[0]:.2e}",
                   epoch_s=round(epoch_s, 1), val_s="")
        msg = f"ep {ep:3d}/{args.epochs} loss {row['loss']:.4f} lr {row['lr']} {fmt_dur(epoch_s)}"
        if va_items and (ep % args.val_every == 0 or ep == args.epochs):
            tv = time.time()
            acc = validate(model, va_items, args.res, dev)
            row.update(val_s=round(time.time() - tv, 1), val_pq=round(acc.pq, 4), val_sq=round(acc.sq, 4),
                       val_dice="", tp=acc.tp, fp=acc.fp, fn=acc.fn, gt_split=acc.gt_split,
                       pred_merge=acc.pred_merge)
            msg += f" | val pq {acc.pq:.4f} sq {acc.sq:.4f} tp {acc.tp} fp {acc.fp} fn {acc.fn} ({fmt_dur(row['val_s'])})"
            if acc.pq > best:
                best = acc.pq
                torch.save({"model": model.state_dict(), "meta": meta}, ckpt)
                msg += " * best"
        elapsed = elapsed0 + time.time() - t_start
        row["elapsed_s"] = round(elapsed, 1)
        tmp = last.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best, "elapsed": elapsed}, tmp)
        tmp.replace(last)
        with open(hist_path, "a", newline="") as fh:
            csv.DictWriter(fh, HISTORY_FIELDS).writerow(row)
        per_epoch = (time.time() - t_start) / len(epoch_times)
        eta = per_epoch * (args.epochs - ep)
        done_at = datetime.now(timezone.utc) + timedelta(seconds=eta)
        msg += f" | elapsed {fmt_dur(elapsed)}, ETA {fmt_dur(eta)} (~{done_at:%H:%M} UTC)"
        print(msg, flush=True)
        print("@@ " + json.dumps({"run": args.name, "epoch": ep, "epochs": args.epochs, "sec_per_epoch": per_epoch,
                                  "best": best}), flush=True)
    if not va_items:
        torch.save({"model": model.state_dict(), "meta": meta}, ckpt)
    (args.out_dir / f"{args.name}.json").write_text(json.dumps(dict(meta, best_val_pq=best), indent=2))
    print("saved", ckpt, "best val pq", best)


if __name__ == "__main__":
    main()
