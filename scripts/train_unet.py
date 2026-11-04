"""Train the U-Net filament segmenter.

Prints progress inside each epoch, an end-of-epoch line with loss, learning rate, timing and ETA,
and val PQ/SQ/pixel Dice every --val-every epochs. Every epoch is also appended to
<out-dir>/<name>.history.csv, and a machine-readable "@@ {json}" line is printed for the Colab
notebook's queue view.

Usage:
  python scripts/train_unet.py --res 512 --epochs 40 --name unet512
  python scripts/train_unet.py --res 1024 --epochs 60 --bs 4 --split all --name unet1024_full
  python scripts/train_unet.py --res 2048 --crop 1024 --targets soft --fold 0 --name effb4_c2048_f0 \
      --encoder efficientnet-b4 --bs 6        # fold 0 of 5, soft targets, native-res crops
  python scripts/train_unet.py --res 1024 --targets soft --fold 0 --encoder efficientnet-b4 --bs 6 \
      --blob-weight 1.0 --fpc-weight 0.3 --ema 0.998 --name effb4_v2_s1024_f0   # recipe v2 (PLAN N3)

Recipe v2 flags (all off by default, and then the run is exactly the plain recipe): --blob-weight adds
the blob (per-instance Dice) loss, --fpc-weight the false-component penalty (filseg.dl), --ema keeps
an exponential moving average of the weights; validation and the saved "model" use the EMA weights.
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

from filseg.data import by_image, encode_masks, fold_split, load_gray, load_train, split  # noqa: E402
from filseg.dl import (TARGET_CHANNELS, FilamentDataset, PostParams, SoftDataset,  # noqa: E402
                       bce_dice_loss, build_model, cached, device, instances, predict_prob, soft_cached,
                       train_loss)
from filseg.metric import PQAccumulator  # noqa: E402


def _union_mask(item, res: int) -> np.ndarray:
    """One annotator's filaments as a boolean mask at res (for the pixel Dice in validation)."""
    import cv2
    from filseg.dl import semantic_mask

    m = semantic_mask(item)
    if res != m.shape[0]:
        m = (cv2.resize(m.astype(np.float32), (res, res), interpolation=cv2.INTER_AREA) >= 0.5)
    return m.astype(bool)


HISTORY_FIELDS = ["epoch", "loss", "lr", "epoch_s", "val_s", "elapsed_s", "val_pq", "val_sq", "val_dice",
                  "tp", "fp", "fn", "gt_split", "pred_merge"]
INSTANCE_FIELDS = ["loss_blob", "loss_fpc"]  # only in the history of runs with an instance loss
# only with --ema: the raw (non-averaged) weights scored too, so an EMA lag can be told apart from the losses;
# checkpoint selection still uses the EMA weights
RAW_FIELDS = ["raw_val_pq", "raw_tp", "raw_fp", "raw_fn"]
# recipe v2 options and their defaults; at the default they stay out of the run's meta
V2_DEFAULTS = {"blob_weight": 0.0, "fpc_weight": 0.0, "fpc_thresh": 0.5, "inst_min_area": 0, "ema": 0.0}


def instance_loss_on(args) -> bool:
    return args.blob_weight > 0 or args.fpc_weight > 0


def resolve_inst_min_area(args) -> int:
    """--inst-min-area, or 0 = 10 px at 1024 scaled with the training input resolution (--res;
    crops keep that scale)."""
    return args.inst_min_area or max(1, round(10 * (args.res / 1024) ** 2))


def run_meta(args) -> dict:
    """The run's arguments for <name>.json and the checkpoint. Recipe v2 options appear only when
    their feature is on, so the meta of a default run is unchanged."""
    meta = {k: v for k, v in vars(args).items()
            if k not in ("out_dir", "resume", "dice_weight") and k not in V2_DEFAULTS}
    if args.dice_weight != 1.0:
        meta["dice_weight"] = args.dice_weight
    if instance_loss_on(args):
        meta.update(blob_weight=args.blob_weight, fpc_weight=args.fpc_weight, fpc_thresh=args.fpc_thresh,
                    inst_min_area=resolve_inst_min_area(args))
    if args.ema > 0:
        meta["ema"] = args.ema
    return meta


def setup_ema(model: torch.nn.Module, decay: float):
    """An EMA copy of the model (weights and BN buffers), or None when decay is 0."""
    if decay <= 0:
        return None
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=True)


def best_state(model: torch.nn.Module, ema, meta: dict) -> dict:
    """Checkpoint contents: "model" is what inference loads, so it holds the EMA weights when EMA is
    on ("model_raw" then keeps the raw weights for a later ablation)."""
    if ema is None:
        return {"model": model.state_dict(), "meta": meta}
    return {"model": ema.module.state_dict(), "model_raw": model.state_dict(), "meta": meta}


def fmt_dur(sec: float) -> str:
    sec = int(max(sec, 0))
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def validate(model, items, res, dev, post: PostParams) -> tuple[PQAccumulator, float]:
    """Quick PQ with fixed cc post-processing, and mean pixel Dice of the thresholded mask;
    one forward pass per image, scored per annotator entry."""
    model.eval()
    acc, dices = PQAccumulator(), []
    preds: dict[str, tuple] = {}
    for it in items:
        if it.image_id not in preds:
            prob = predict_prob(model, load_gray(it.path), res, dev, tta=False)[0]
            preds[it.image_id] = (encode_masks(instances(prob, post)), prob > post.thresh)
        rles, fg = preds[it.image_id]
        acc.add(it.gt_rles, rles)
        gt = _union_mask(it, res)
        dices.append(2 * (fg & gt).sum() / max(fg.sum() + gt.sum(), 1))
    model.train()
    return acc, float(np.mean(dices))


def build_parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--fold", type=int, default=-1,
                    help="0..4: train on the other four folds, validate on this one (overrides --split)")
    ap.add_argument("--crop", type=int, default=0,
                    help="random crop size for --targets soft (e.g. 1024 with --res 2048); 0 = whole image")
    ap.add_argument("--dice-weight", type=float, default=1.0,
                    help="weight of the soft Dice term next to BCE (0 = BCE only: calibrated soft maps)")
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="unet")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/models"))
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out-dir>/<name>.last.pt (written after every epoch)")
    # recipe v2 (docs/PLAN.md N3); defaults reproduce the plain recipe exactly
    ap.add_argument("--blob-weight", type=float, default=V2_DEFAULTS["blob_weight"],
                    help="weight of the blob (per-instance Dice) loss; needs --targets soft")
    ap.add_argument("--fpc-weight", type=float, default=V2_DEFAULTS["fpc_weight"],
                    help="weight of the false-component loss (predicted blobs touching no target); "
                         "needs --targets soft")
    ap.add_argument("--fpc-thresh", type=float, default=V2_DEFAULTS["fpc_thresh"],
                    help="probability threshold that defines predicted components for --fpc-weight")
    ap.add_argument("--inst-min-area", type=int, default=V2_DEFAULTS["inst_min_area"],
                    help="smallest instance/component (px at --res) the instance losses score; "
                         "0 = 10 px at 1024, scaled")
    ap.add_argument("--ema", type=float, default=V2_DEFAULTS["ema"],
                    help="EMA decay of the weights (e.g. 0.998); validation and the saved model use the EMA; 0 = off")
    return ap


def parse_args(argv=None) -> argparse.Namespace:
    ap = build_parser()
    args = ap.parse_args(argv)
    if instance_loss_on(args) and args.targets != "soft":
        ap.error("--blob-weight / --fpc-weight need --targets soft")
    return args


def main() -> None:
    args = parse_args()

    torch.manual_seed(0)
    dev = device()
    data = load_train()
    if args.fold >= 0:
        tr_items, va_items = fold_split(data, args.fold)
    else:
        tr_items = split(data, args.split)
        va_items = split(data, "val") if args.split == "train" else []
    print(f"device={dev} train={len(tr_items)} val={len(va_items)}", flush=True)
    if args.targets == "soft":
        # one sample per image, drawn in proportion to its annotator count (as the metric weights it)
        groups = list(by_image(tr_items).values())
        for g in groups:  # build cache up front (single process, avoids worker races)
            soft_cached(g, args.res)
        if instance_loss_on(args):
            dataset = SoftDataset(groups, args.res, augment=True, crop=args.crop, instances=True,
                                  inst_min_area=resolve_inst_min_area(args))
        else:
            dataset = SoftDataset(groups, args.res, augment=True, crop=args.crop)
        sampler = torch.utils.data.WeightedRandomSampler([float(len(g)) for g in groups],
                                                         num_samples=len(tr_items), replacement=True)
        loader = DataLoader(dataset, batch_size=args.bs, sampler=sampler, num_workers=args.workers,
                            drop_last=True, persistent_workers=args.workers > 0)
    else:
        for it in tr_items:  # build cache up front (single process, avoids worker races)
            cached(it, args.res)
        loader = DataLoader(FilamentDataset(tr_items, args.res, augment=True, targets=args.targets),
                            batch_size=args.bs, shuffle=True, num_workers=args.workers, drop_last=True,
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
    meta = run_meta(args)
    inst_loss = instance_loss_on(args)
    min_area = resolve_inst_min_area(args)
    fields = HISTORY_FIELDS + (INSTANCE_FIELDS if inst_loss else []) + (RAW_FIELDS if args.ema > 0 else [])
    hist_path = args.out_dir / f"{args.name}.history.csv"
    best, start, elapsed0 = -1.0, 1, 0.0
    if args.resume and last.exists():
        st = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        sched.total_steps = steps  # allows resuming with a larger --epochs
        scaler.load_state_dict(st["scaler"])
        best, start = st["best"], st["epoch"] + 1
        elapsed0 = st.get("elapsed", 0.0)
        print(f"resumed from {last} at epoch {start} (best val pq {best:.4f})", flush=True)
    # EMA of the weights (after a resume: restored, or restarted from the resumed weights)
    ema = setup_ema(model, args.ema)
    if ema is not None and start > 1:
        if "ema" in st:
            ema.load_state_dict(st["ema"])
            print("EMA state restored", flush=True)
        else:
            print("no EMA state in the resume file: EMA restarts from the resumed weights", flush=True)
    eval_model = ema.module if ema is not None else model
    if start == 1 or not hist_path.exists():
        with open(hist_path, "w", newline="") as fh:
            csv.writer(fh).writerow(fields)
    # soft targets predict the annotator fraction, so their probabilities run lower
    post = PostParams(thresh=0.35 if args.targets == "soft" else 0.5)
    n_steps = len(loader)
    report_every = max(1, n_steps // 4)
    t_start, epoch_times = time.time(), []
    print(f"{args.name}: epochs {start}..{args.epochs}, {n_steps} steps/epoch, batch {args.bs}, "
          f"res {args.res}, encoder {args.encoder}, targets {args.targets}", flush=True)
    for ep in range(start, args.epochs + 1):
        t0, losses, parts = time.time(), [], []
        for step, batch in enumerate(loader, 1):
            x, y = batch[0].to(dev, non_blocking=True), batch[1].to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            if inst_loss:
                loss, comp = train_loss(logits.float(), y, batch[2].to(dev, non_blocking=True), args.blob_weight,
                                        args.fpc_weight, args.fpc_thresh, min_area)
                parts.append((comp["blob"], comp["fpc"]))
            else:
                loss = bce_dice_loss(logits.float(), y, args.dice_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update_parameters(model)
            sched.step()
            losses.append(loss.item())
            if step % report_every == 0 and step < n_steps:
                dt = time.time() - t0
                print(f"  ep {ep}/{args.epochs} step {step}/{n_steps} loss {np.mean(losses):.4f} "
                      f"{step / dt:.2f} it/s, epoch done in ~{fmt_dur(dt / step * (n_steps - step))}", flush=True)
        epoch_s = time.time() - t0
        epoch_times.append(epoch_s)
        row = dict(epoch=ep, loss=round(float(np.mean(losses)), 5), lr=f"{sched.get_last_lr()[0]:.2e}",
                   epoch_s=round(epoch_s, 1), val_s="")
        msg = f"ep {ep:3d}/{args.epochs} loss {row['loss']:.4f} lr {row['lr']} {fmt_dur(epoch_s)}"
        if inst_loss:
            blob, fpc = np.mean(parts, axis=0)
            row.update(loss_blob=round(float(blob), 5), loss_fpc=round(float(fpc), 5))
            msg += f" (blob {blob:.4f} fpc {fpc:.4f})"
        if va_items and (ep % args.val_every == 0 or ep == args.epochs):
            tv = time.time()
            acc, dice = validate(eval_model, va_items, args.res, dev, post)
            row.update(val_s=round(time.time() - tv, 1), val_pq=round(acc.pq, 4), val_sq=round(acc.sq, 4),
                       val_dice=round(dice, 4), tp=acc.tp, fp=acc.fp, fn=acc.fn, gt_split=acc.gt_split,
                       pred_merge=acc.pred_merge)
            msg += (f" | val pq {acc.pq:.4f} sq {acc.sq:.4f} dice {dice:.4f} tp {acc.tp} fp {acc.fp} fn {acc.fn}"
                    f" ({fmt_dur(row['val_s'])})")
            if ema is not None:
                raw, _ = validate(model, va_items, args.res, dev, post)
                row.update(raw_val_pq=round(raw.pq, 4), raw_tp=raw.tp, raw_fp=raw.fp, raw_fn=raw.fn)
                msg += f" | raw pq {raw.pq:.4f} tp {raw.tp} fp {raw.fp}"
            if acc.pq > best:
                best = acc.pq
                torch.save(best_state(model, ema, meta), ckpt)
                msg += " * best"
        elapsed = elapsed0 + time.time() - t_start
        row["elapsed_s"] = round(elapsed, 1)
        # full training state, written atomically so a session cut mid-write leaves the previous one
        tmp = last.with_suffix(".tmp")
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "scaler": scaler.state_dict(), "epoch": ep, "best": best, "elapsed": elapsed}
        if ema is not None:
            state["ema"] = ema.state_dict()
        torch.save(state, tmp)
        tmp.replace(last)
        with open(hist_path, "a", newline="") as fh:
            csv.DictWriter(fh, fields).writerow(row)
        # ETA from the mean epoch time of this process (validation included)
        per_epoch = (time.time() - t_start) / len(epoch_times)
        eta = per_epoch * (args.epochs - ep)
        done_at = datetime.now(timezone.utc) + timedelta(seconds=eta)
        msg += f" | elapsed {fmt_dur(elapsed)}, ETA {fmt_dur(eta)} (~{done_at:%H:%M} UTC)"
        print(msg, flush=True)
        print("@@ " + json.dumps({"run": args.name, "epoch": ep, "epochs": args.epochs, "sec_per_epoch": per_epoch,
                                  "best": best}), flush=True)
    if not va_items:
        torch.save(best_state(model, ema, meta), ckpt)
    (args.out_dir / f"{args.name}.json").write_text(json.dumps(dict(meta, best_val_pq=best), indent=2))
    print("saved", ckpt, "best val pq", best)


if __name__ == "__main__":
    main()
