"""Error analysis of one pipeline on held-out maps: per annotator and annotator group, by how many
other annotators drew the same filament, by filament size; rubric-style metrics (per-entry pixel
Dice, 1:n / n:1 overlap rates); and the inter-annotator PQ on all multi-annotator train images.

The annotator id is the prefix of the annotation image id (`<annotator>-<stem>`); its first two
digits look like a labelling group.

Usage:
  python scripts/analyze_errors.py outputs/colab/unet1024_effb4/probs_val configs/post_unet1024_effb4_hyst.json
"""
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filseg.data import by_image, encode_masks, load_train, with_maps  # noqa: E402
from filseg.dl import PostParams, instances_lowres, load_maps, upsample_labels  # noqa: E402

PROBS = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "outputs/colab/unet1024_effb4/probs_val")
POST = PostParams(**json.loads(Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "configs/post_unet1024_effb4_hyst.json").read_text()))


def iou(a, b):
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return np.asarray(mu.iou(a, b, [0] * len(b)))  # (len(a), len(b))


def pq_parts(m):
    """m: (n_gt, n_pred) iou. returns sumiou, tp, fp, fn, matched gt mask, matched pred idx"""
    hit = m > 0.5
    return m[hit].sum(), hit.sum(), (hit.sum(0) == 0).sum(), (hit.sum(1) == 0).sum()


def pq(s):
    d = s[1] + .5 * s[2] + .5 * s[3]
    return s[0] / d if d else 0


def job(args):
    stem, entries = args
    prob, group = load_maps(Path(PROBS), stem)
    inst = instances_lowres(prob, POST, group)
    pred = encode_masks(upsample_labels(inst))
    parea = mu.area(pred) if pred else np.zeros(0)
    fg = upsample_labels(inst) > 0
    out = []
    for key, gts in entries:
        m = iou(gts, pred)
        g = np.zeros_like(fg)
        for r in gts:
            g |= mu.decode(r).astype(bool)
        dice = 2 * (g & fg).sum() / max(g.sum() + fg.sum(), 1)
        ov = m > 0.1  # 1:n = a gt touched by > 1 prediction; n:1 = a prediction touching > 1 gt
        garea = mu.area(gts) if gts else np.zeros(0)
        # support: how many OTHER annotators of this image have a match > 0.5 for this gt
        others = [g for k, g in entries if k != key]
        sup = np.zeros(len(gts), int)
        for og in others:
            if gts and og:
                sup += (iou(gts, og).max(1) > 0.5)
        best = m.max(1) if m.size else np.zeros(len(gts))
        arg = m.argmax(1) if m.size else np.zeros(len(gts), int)
        ratio = np.array([parea[arg[i]] / garea[i] if best[i] > 0.5 else np.nan for i in range(len(gts))])
        # FN type: no overlap / near (0.3-0.5) / covered by a pred that also matches... (merged)
        out.append(dict(key=key, stem=stem, n_ann=len(entries), parts=pq_parts(m), garea=garea, best=best,
                        sup=sup, ratio=ratio, npred=len(pred), ngt=len(gts),
                        pred_hits=(m > 0.5).sum(0) if m.size else np.zeros(len(pred), int),
                        dice=dice, one_n=int((ov.sum(1) > 1).sum()), n_one=int((ov.sum(0) > 1).sum())))
    return out


def human(args):
    """each annotator scored against each other annotator of the image (as 'prediction')."""
    stem, entries = args
    res = []
    for k1, g1 in entries:
        for k2, g2 in entries:
            if k1 != k2:
                res.append((k1.split('-')[0], k2.split('-')[0], pq_parts(iou(g1, g2))))
    return res


def main():
    items = load_train()
    val = with_maps(items, Path(PROBS))
    groups = [(s, [(t.key, t.gt_rles) for t in es]) for s, es in by_image(val).items()]
    allg = [(s, [(t.key, t.gt_rles) for t in es]) for s, es in by_image(items).items() if len(es) > 1]
    with ProcessPoolExecutor() as ex:
        rows = [r for rs in ex.map(job, groups) for r in rs]
        hum = [r for rs in ex.map(human, allg) for r in rs]
    tot = np.sum([r['parts'] for r in rows], 0)
    print(f"model PQ {pq(tot):.4f} TP {tot[1]} FP {tot[2]} FN {tot[3]} SQ {tot[0]/tot[1]:.3f}")

    print(f"mean pixel Dice per entry {np.mean([r['dice'] for r in rows]):.3f}  "
          f"1:n rate {sum(r['one_n'] for r in rows) / max(sum(r['ngt'] for r in rows), 1):.3f}  "
          f"n:1 rate {sum(r['n_one'] for r in rows) / max(sum(r['npred'] for r in rows), 1):.3f}")

    print("\n== by annotator group (first 2 digits of the annotator id) ==")
    grp = defaultdict(list)
    for r in rows:
        grp[r['key'][:2]].append(r['parts'])
    for g, v in sorted(grp.items()):
        print(f"  group {g}: entries {len(v):3d} PQ {pq(np.sum(v, 0)):.3f}")

    print("\n== by #annotators of image ==")
    for n in (1, 2, 3):
        s = np.sum([r['parts'] for r in rows if r['n_ann'] == n], 0)
        print(f"  n_ann={n}: entries {sum(r['n_ann']==n for r in rows)} PQ {pq(s):.4f} TP {s[1]} FP {s[2]} FN {s[3]}")

    print("\n== per annotator (val, >=4 entries) ==  PQ | gt/entry | pred/img | recall | prec | matched area ratio")
    per = defaultdict(list)
    for r in rows:
        per[r['key'].split('-')[0]].append(r)
    lines = []
    for a, rs in per.items():
        s = np.sum([r['parts'] for r in rs], 0)
        ratio = np.nanmedian(np.concatenate([r['ratio'] for r in rs])) if s[1] else np.nan
        lines.append((pq(s), a, len(rs), np.mean([r['ngt'] for r in rs]), np.mean([r['npred'] for r in rs]),
                      s[1] / max(s[1] + s[3], 1), s[1] / max(s[1] + s[2], 1), ratio))
    for l in sorted(lines):
        if l[2] >= 4:
            print(f"  {l[1]} n={l[2]:3d} PQ {l[0]:.3f} gt {l[3]:4.1f} pred {l[4]:4.1f} rec {l[5]:.2f} prec {l[6]:.2f} ratio {l[7]:.2f}")
    pqs = np.array([l[0] for l in lines if l[2] >= 4])
    print(f"  spread of per-annotator PQ: min {pqs.min():.3f} p25 {np.percentile(pqs,25):.3f} med {np.median(pqs):.3f} p75 {np.percentile(pqs,75):.3f} max {pqs.max():.3f}")

    print("\n== recall by GT support (multi-annotator images; support = #other annotators matching it) ==")
    for n in (2, 3):
        for k in range(n):
            b = np.concatenate([r['best'][r['sup'] == k] for r in rows if r['n_ann'] == n])
            print(f"  n_ann={n} support={k}: {len(b):4d} gts, recall {np.mean(b>0.5):.2f}, near-miss(0.3-0.5) {np.mean((b>0.3)&(b<=0.5)):.2f}")

    print("\n== recall by GT area (px @2048) ==")
    ga = np.concatenate([r['garea'] for r in rows]); be = np.concatenate([r['best'] for r in rows])
    rat = np.concatenate([r['ratio'] for r in rows])
    for lo, hi in [(0, 400), (400, 800), (800, 1600), (1600, 3200), (3200, 8000), (8000, 1e9)]:
        sel = (ga >= lo) & (ga < hi)
        print(f"  [{lo:5.0f},{hi:7.0f}) n={sel.sum():4d} recall {np.mean(be[sel]>0.5):.2f} near {np.mean((be[sel]>0.3)&(be[sel]<=0.5)):.2f} "
              f"none(<0.1) {np.mean(be[sel]<0.1):.2f} meanIoU|TP {be[sel][be[sel]>0.5].mean():.3f} pred/gt area {np.nanmedian(rat[sel]):.2f}")

    print("\n== FP predictions: how many annotators of the image match each prediction ==")
    for n in (1, 2, 3):
        imgs = {}
        for r in rows:
            if r['n_ann'] == n:
                imgs.setdefault(r['stem'], []).append(r['pred_hits'] > 0)
        cnt = np.concatenate([np.sum(v, 0) for v in imgs.values()]) if imgs else np.zeros(0)
        print(f"  n_ann={n}: preds {len(cnt)}, matched by k annotators: " + ", ".join(f"{k}:{np.mean(cnt==k):.2f}" for k in range(n + 1)))

    print("\n== human agreement (all train multi-annotator images): annotator A scored as prediction vs B ==")
    s = np.sum([h[2] for h in hum], 0)
    print(f"  pooled PQ {pq(s):.4f}  SQ {s[0]/s[1]:.3f}  TP {s[1]} FP {s[2]} FN {s[3]}")
    pa = defaultdict(list)
    for a, b, p in hum:
        pa[a].append(p)
    ls = sorted((pq(np.sum(v, 0)), a, len(v)) for a, v in pa.items() if len(v) >= 10)
    print("  annotator-as-predictor PQ (vs the others):", ", ".join(f"{a}:{p:.2f}" for p, a, _ in ls))


if __name__ == "__main__":
    main()
