"""Decomposition of the PQ gap of one OOF pipeline (research05). Diagnostics only, nothing tuned.

For every annotator entry (an annotator-image, scored separately by the host) the pipeline's
predictions are matched to the entry's filaments with the metric in src/filseg/metric.py, and each
error is classified:

1. PQ / SQ / RQ / TP / FP / FN, exact "PQ lost" split (1 - PQ = boundary + FP + FN shares), and
   upper bounds: (a) SQ = 1; (b) FN with best IoU in (0.3, 0.5] become TP at that IoU (their best
   prediction, if FP, stops being FP; greedy one pred per GT); (c) no FP; (d) split / merge fixed:
   split = FN GT covered by >= 2 fragment predictions (> 50 % of each inside the GT) whose union IoU
   > 0.5; merge = a prediction that covers >= 2 GTs of the entry (> 50 % of each GT inside it), cut
   into one piece per GT, extra pixels shared in proportion, pieces with IoU > 0.5 become TP.
2. FN by best IoU, by GT area, by annotator support (how many other annotators of the image drew a
   filament matching it with IoU > 0.5, and the share of its pixels covered by any other annotator).
3. FP by kind (no annotator's GT at all / matches another annotator's filament / overlaps this
   entry's GT partially / overlaps only other annotators' GT partially), by area, by peak prob.
4. TP area ratio pred / GT, and global per-instance erode / dilate by 1-2 px at 2048 (full PQ).
5. Inter-annotator PQ: each annotator scored against each other annotator of the same image.

Usage:
  python scripts/research/error_decomp.py --probs outputs/ens/n1n2_oof \
      --post configs/post_ens_n1n2_hyst.json --jobs 4 > docs/logs/research05/error_decomp.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filseg.data import by_image, encode_masks, load_train, with_maps  # noqa: E402
from filseg.dl import PostParams, instances_lowres, load_maps, upsample_labels  # noqa: E402
from filseg.metric import IOU_THRESHOLD, PQAccumulator  # noqa: E402

PROBS: Path
POST: PostParams
MORPH = (-2, -1, 1, 2)   # erode (<0) / dilate (>0) radii in px at 2048


def _disk(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))


def morph(labels: np.ndarray, k: int) -> np.ndarray:
    f = labels.astype(np.float32)
    if k > 0:   # grow every instance into background only (max label wins at contacts)
        grown = cv2.dilate(f, _disk(k))
        return np.where(labels > 0, labels, grown).astype(np.int32)
    d = _disk(-k)   # keep pixels whose whole disk has the same label
    keep = (cv2.erode(f, d) == f) & (cv2.dilate(f, d) == f) & (labels > 0)
    return np.where(keep, labels, 0).astype(np.int32)


def iou_mat(gts: list[dict], preds: list[dict]) -> np.ndarray:
    if not gts or not preds:
        return np.zeros((len(gts), len(preds)))
    return np.asarray(mu.iou(preds, gts, [0] * len(gts))).T


def entry_analysis(gts, labels, n_pred, parea, others):
    """gts: this entry's RLEs; labels: pred label image; others: list of other annotators' RLE lists."""
    n_gt = len(gts)
    garea = np.array([mu.area(g) for g in gts], float)
    inter = np.zeros((n_gt, n_pred))
    for i, g in enumerate(gts):
        m = mu.decode(g).astype(bool)
        bc = np.bincount(labels[m], minlength=n_pred + 1)[1:n_pred + 1]
        inter[i] = bc
    union = garea[:, None] + parea[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0)
    hit = iou > IOU_THRESHOLD
    gt_tp = hit.any(1)
    pred_tp = hit.any(0)
    tp_iou = iou[hit].tolist()
    tp_pairs = [(float(parea[j] / garea[i]), float(garea[i]), float(iou[i, j])) for i, j in zip(*np.nonzero(hit))]
    best = iou.max(1) if n_pred else np.zeros(n_gt)
    best_j = iou.argmax(1) if n_pred else np.zeros(n_gt, int)

    # (b) near misses
    used = set()
    nm_gain, nm_fp_drop, nm_n = 0.0, 0, 0
    order = np.argsort(-best)
    for i in order:
        if gt_tp[i] or not (0.3 < best[i] <= 0.5):
            continue
        j = int(best_j[i])
        if j in used:
            continue
        used.add(j)
        nm_n += 1
        nm_gain += best[i]
        if not pred_tp[j]:
            nm_fp_drop += 1

    # (d) split: FN GT with >= 2 fragments (>50% of pred inside the GT), union IoU > 0.5
    frag = (inter / np.maximum(parea[None, :], 1)) > 0.5
    split_n, split_gain, split_fp_drop = 0, 0.0, set()
    for i in range(n_gt):
        if gt_tp[i]:
            continue
        js = np.nonzero(frag[i])[0]
        if len(js) < 2:
            continue
        ui = inter[i, js].sum() / (garea[i] + parea[js].sum() - inter[i, js].sum())
        if ui > IOU_THRESHOLD:
            split_n += 1
            split_gain += ui
            split_fp_drop.update(int(j) for j in js if not pred_tp[j])
    # merge: pred covering >= 2 GTs (>50% of each GT inside the pred)
    cov = (inter / np.maximum(garea[:, None], 1)) > 0.5
    # merge deltas: (preds, dS, dTP, dFN, dFP)
    merge_n_pred, m_ds, m_dtp, m_dfp = 0, 0.0, 0, 0
    for j in range(n_pred):
        gi = np.nonzero(cov[:, j])[0]
        if len(gi) < 2:
            continue
        merge_n_pred += 1
        extra = parea[j] - inter[gi, j].sum()
        share = inter[gi, j] / inter[gi, j].sum()
        piece = inter[gi, j] / (garea[gi] + extra * share)
        good = piece > IOU_THRESHOLD
        for g, pv, ok in zip(gi, piece, good):
            if hit[g, j]:            # existing TP of this pred: IoU changes, or the TP is lost
                m_ds += (pv if ok else 0.0) - iou[g, j]
                m_dtp -= 0 if ok else 1
            elif ok and not gt_tp[g]:
                m_ds += pv
                m_dtp += 1
        if not pred_tp[j] and good.any():
            m_dfp -= 1

    # annotator support of each GT
    support, other_cov = [], []
    for i, g in enumerate(gts):
        if not others:
            support.append(-1)
            other_cov.append(-1.0)
            continue
        s = 0
        for o in others:
            if o and iou_mat([g], o).max() > IOU_THRESHOLD:
                s += 1
        support.append(s)
        allo = [r for o in others for r in o]
        if allo:
            u = mu.merge(allo)
            other_cov.append(float(mu.area(mu.merge([g, u], intersect=True)) / max(garea[i], 1)))
        else:
            other_cov.append(0.0)
    gt_rows = [dict(tp=bool(gt_tp[i]), best=float(best[i]), area=float(garea[i]), support=support[i],
                    k=len(others) + 1, other_cov=other_cov[i]) for i in range(n_gt)]
    return dict(tp_iou=tp_iou, tp_pairs=tp_pairs, n_gt=n_gt, gt_rows=gt_rows, pred_tp=pred_tp.tolist(),
                inter_this=(inter.sum(0) / np.maximum(parea, 1)).tolist(),
                nm=(nm_n, nm_gain, nm_fp_drop),
                split=(split_n, split_gain, len(split_fp_drop)),
                merge=(merge_n_pred, m_ds, m_dtp, m_dfp))


def image_job(args):
    stem, entries = args
    prob, group = load_maps(PROBS, stem)
    labels = upsample_labels(instances_lowres(prob, POST, group))
    if prob.shape[0] != labels.shape[0]:
        prob = cv2.resize(prob, labels.shape[::-1], interpolation=cv2.INTER_LINEAR)
    n_pred = int(labels.max())
    parea = np.bincount(labels.ravel(), minlength=n_pred + 1)[1:].astype(float)
    peak = np.zeros(n_pred + 1, np.float32)
    np.maximum.at(peak, labels.ravel(), prob.ravel())
    peak = peak[1:]
    preds = encode_masks(labels)
    morphed = {k: encode_masks(morph(labels, k)) for k in MORPH}
    out = []
    all_gts = [g for _, gs in entries for g in gs]
    for ei, (key, gts) in enumerate(entries):
        others = [gs for ej, (_, gs) in enumerate(entries) if ej != ei]
        r = entry_analysis(gts, labels, n_pred, parea, others)
        # FP kinds
        other_rles = [g for o in others for g in o]
        iou_o = iou_mat(other_rles, preds)                       # (n_other_gt, n_pred)
        iou_all = iou_mat(all_gts, preds)
        fp_rows = []
        for j in range(n_pred):
            if r["pred_tp"][j]:
                continue
            if iou_o.size and iou_o[:, j].max() > IOU_THRESHOLD:
                kind = "match_other_annot"
            elif r["inter_this"][j] > 0:
                kind = "partial_this_gt"
            elif iou_all.size and iou_all[:, j].max() > 0:
                kind = "partial_other_gt"
            else:
                kind = "no_gt_any_annot"
            fp_rows.append(dict(kind=kind, area=float(parea[j]), peak=float(peak[j]),
                                this_cov=float(r["inter_this"][j])))
        r["fp_rows"] = fp_rows
        r["k"] = len(entries)
        # morphology variants (full PQ recomputed)
        r["morph"] = {}
        for k in MORPH:
            p2 = morphed[k]
            iou2 = iou_mat(gts, p2)
            h2 = iou2 > IOU_THRESHOLD
            r["morph"][k] = (iou2[h2].tolist(), int((h2.sum(0) == 0).sum()) if len(p2) else 0,
                             int((h2.sum(1) == 0).sum()) if len(gts) else 0)
        del r["pred_tp"], r["inter_this"]
        out.append((key, gts, preds, r))
    # inter-annotator pairs
    inter_pairs = []
    for ei, (_, ge) in enumerate(entries):
        for ej, (_, gp) in enumerate(entries):
            if ei != ej:
                iou = iou_mat(ge, gp)
                h = iou > IOU_THRESHOLD
                inter_pairs.append((iou[h].tolist(), int((h.sum(0) == 0).sum()) if gp else 0,
                                    int((h.sum(1) == 0).sum()) if ge else 0))
    return out, inter_pairs


def init(probs: Path, post: PostParams) -> None:
    global PROBS, POST
    PROBS, POST = probs, post
    cv2.setNumThreads(1)


def pq(s, tp, fp, fn):
    d = tp + 0.5 * fp + 0.5 * fn
    return s / d if d else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", type=Path, default=ROOT / "outputs/ens/n1n2_oof")
    ap.add_argument("--post", type=Path, default=ROOT / "configs/post_ens_n1n2_hyst.json")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    post = PostParams(**json.loads(args.post.read_text()))
    groups = [(s, [(t.key, t.gt_rles) for t in es])
              for s, es in by_image(with_maps(load_train(), args.probs)).items()]
    with ProcessPoolExecutor(args.jobs, initializer=init, initargs=(args.probs, post)) as ex:
        res = list(ex.map(image_job, groups, chunksize=4))
    entries = [e for o, _ in res for e in o]
    pairs = [p for _, ps in res for p in ps]
    P = print
    P(f"probs {args.probs}\npost {post.to_dict()}\nimages {len(groups)}  entries {len(entries)}")

    # sanity: metric module
    acc = PQAccumulator()
    for _, g, p, _ in entries:
        acc.add(g, p)
    rows = [e[3] for e in entries]
    tpi = np.concatenate([r["tp_iou"] for r in rows])
    S, TP = tpi.sum(), len(tpi)
    gtr = [g for r in rows for g in r["gt_rows"]]
    fpr = [f for r in rows for f in r["fp_rows"]]
    FN = sum(not g["tp"] for g in gtr)
    FP = len(fpr)
    base = pq(S, TP, FP, FN)
    D = TP + 0.5 * FP + 0.5 * FN
    P("\n== 1. Overall ==")
    P(f"metric.py: {acc.summary()}")
    P(f"PQ {base:.4f}  SQ {S / TP:.4f}  RQ {TP / D:.4f}  TP {TP}  FP {FP}  FN {FN}  GT {TP + FN}")
    P(f"FP/TP {FP / TP:.3f}  FN/TP {FN / TP:.3f}")
    lost = 1 - base
    P(f"1-PQ = {lost:.4f}: boundary (TP-sumIoU)/D {(TP - S) / D:.4f} ({(TP - S) / D / lost:.0%}), "
      f"FP 0.5FP/D {0.5 * FP / D:.4f} ({0.5 * FP / D / lost:.0%}), FN 0.5FN/D {0.5 * FN / D:.4f} ({0.5 * FN / D / lost:.0%})")
    nm = np.array([r["nm"] for r in rows]).sum(0)
    sp = np.array([r["split"] for r in rows]).sum(0)
    mg = np.array([r["merge"] for r in rows]).sum(0)
    P("Upper bounds (each alone, from the base):")
    P(f" (a) SQ=1                                   PQ {TP / D:.4f}  (+{TP / D - base:.4f})")
    b = pq(S + nm[1], TP + nm[0], FP - nm[2], FN - nm[0])
    P(f" (b) FN with best IoU (0.3,0.5] -> TP       PQ {b:.4f}  (+{b - base:.4f})  [{int(nm[0])} GTs, {int(nm[2])} FP removed]")
    c = pq(S, TP, 0, FN)
    P(f" (c) all FP removed                         PQ {c:.4f}  (+{c - base:.4f})")
    for name, n in (("no_gt_any_annot", None), ("match_other_annot", None), ("partial_this_gt", None),
                    ("partial_other_gt", None)):
        nf = sum(f["kind"] == name for f in fpr)
        v = pq(S, TP, FP - nf, FN)
        P(f"     (c.{name:18s}) only: {nf:5d} FP     PQ {v:.4f}  (+{v - base:.4f})")
    d1 = pq(S + sp[1], TP + sp[0], FP - sp[2], FN - sp[0])
    P(f" (d1) split fixed                           PQ {d1:.4f}  (+{d1 - base:.4f})  [{int(sp[0])} GTs, {int(sp[2])} FP fragments removed]")
    d2 = pq(S + mg[1], TP + mg[2], FP + mg[3], FN - mg[2])
    P(f" (d2) merge fixed                           PQ {d2:.4f}  (+{d2 - base:.4f})  [{int(mg[0])} merging preds, net {int(mg[2]):+d} TP, {int(mg[3]):+d} FP]")
    d = pq(S + sp[1] + mg[1], TP + sp[0] + mg[2], FP - sp[2] + mg[3], FN - sp[0] - mg[2])
    P(f" (d) split+merge fixed                      PQ {d:.4f}  (+{d - base:.4f})")
    P(f" counts: gt_split(touched by >1 pred) {acc.gt_split}, pred_merge(touches >1 GT) {acc.pred_merge}")

    # per k
    P("\nBy number of annotators k of the image (model PQ on those entries):")
    for k in sorted({r["k"] for r in rows}):
        rr = [r for r in rows if r["k"] == k]
        t = np.concatenate([r["tp_iou"] for r in rr]) if rr else np.zeros(0)
        fn_ = sum(not g["tp"] for r in rr for g in r["gt_rows"])
        fp_ = sum(len(r["fp_rows"]) for r in rr)
        P(f" k={k}: entries {len(rr):4d}  PQ {pq(t.sum(), len(t), fp_, fn_):.4f}  SQ {t.mean():.4f}  TP {len(t)} FP {fp_} FN {fn_}")

    # 2. FN
    P("\n== 2. FN breakdown ==")
    fn_rows = [g for g in gtr if not g["tp"]]
    for lo, hi, lab in ((-1, 0, "best IoU = 0 (fully missed)"), (0, 0.3, "(0, 0.3]"), (0.3, 0.5, "(0.3, 0.5]")):
        n = sum(lo < g["best"] <= hi for g in fn_rows) if lo >= 0 else sum(g["best"] == 0 for g in fn_rows)
        P(f" {lab:28s} {n:5d}  ({n / FN:.1%})  PQ if fixed at IoU 0.5 (no FP change) "
          f"{pq(S + 0.5 * n, TP + n, FP, FN - n) - base:+.4f}")
    abins = [0, 1000, 3000, 10000, 30000, 1e9]
    P(" by GT area (px at 2048): bin  GT  FN  FN-rate  share-of-FN  TP-SQ")
    for lo, hi in zip(abins[:-1], abins[1:]):
        gg = [g for g in gtr if lo <= g["area"] < hi]
        f = sum(not g["tp"] for g in gg)
        sq = [p[2] for r in rows for p in r["tp_pairs"] if lo <= p[1] < hi]
        P(f"  [{lo:>6.0f},{hi:>8.0f})  {len(gg):5d} {f:5d}  {f / max(len(gg), 1):.3f}  {f / FN:.3f}  "
          f"{np.mean(sq) if sq else 0:.3f}")
    P(" by annotator support (k>=2 images; support = other annotators with an IoU>0.5 matching filament):")
    for k in sorted({g["k"] for g in gtr if g["k"] > 1}):
        for s in range(k):
            gg = [g for g in gtr if g["k"] == k and g["support"] == s]
            f = sum(not g["tp"] for g in gg)
            P(f"  k={k} support={s}/{k - 1}: GT {len(gg):5d}  FN {f:5d}  FN-rate {f / max(len(gg), 1):.3f}  share-of-FN {f / FN:.3f}")
    gg = [g for g in gtr if g["k"] == 1]
    f = sum(not g["tp"] for g in gg)
    P(f"  k=1 (single annotator): GT {len(gg):5d}  FN {f:5d}  FN-rate {f / max(len(gg), 1):.3f}  share-of-FN {f / FN:.3f}")
    solo = [g for g in fn_rows if g["k"] > 1 and g["support"] == 0]
    nocov = [g for g in fn_rows if g["k"] > 1 and g["other_cov"] < 0.1]
    P(f" FN drawn by only this annotator (support 0, k>=2): {len(solo)} ({len(solo) / FN:.1%} of FN); "
      f"of those with <10% pixels covered by any other annotator: {len(nocov)} ({len(nocov) / FN:.1%} of FN)")
    P(f"   their best IoU: =0 {sum(g['best'] == 0 for g in solo)}, (0,.3] {sum(0 < g['best'] <= .3 for g in solo)}, "
      f"(.3,.5] {sum(.3 < g['best'] <= .5 for g in solo)}")
    sup = [g for g in fn_rows if g["k"] > 1 and g["support"] > 0]
    P(f" FN agreed by >=1 other annotator: {len(sup)} ({len(sup) / FN:.1%} of FN); best IoU =0 {sum(g['best'] == 0 for g in sup)}, "
      f"(0,.3] {sum(0 < g['best'] <= .3 for g in sup)}, (.3,.5] {sum(.3 < g['best'] <= .5 for g in sup)}")
    v = pq(S, TP, FP, FN - len(solo))
    P(f" PQ if solo-annotator FN were not counted: {v:.4f} ({v - base:+.4f})")

    # 3. FP
    P("\n== 3. FP breakdown ==")
    kinds = ["no_gt_any_annot", "match_other_annot", "partial_this_gt", "partial_other_gt"]
    for kd in kinds:
        ff = [f for f in fpr if f["kind"] == kd]
        P(f" {kd:18s} {len(ff):5d} ({len(ff) / FP:.1%})  median area {np.median([f['area'] for f in ff]) if ff else 0:7.0f}  "
          f"median peak {np.median([f['peak'] for f in ff]) if ff else 0:.3f}")
    ff = [f for f in fpr if f["kind"] == "partial_this_gt"]
    if ff:
        c = np.array([f["this_cov"] for f in ff])
        P(f"   partial_this_gt: share of pred inside this entry's GT: <0.25 {np.mean(c < .25):.2f}, "
          f"0.25-0.5 {np.mean((c >= .25) & (c < .5)):.2f}, >=0.5 {np.mean(c >= .5):.2f}")
    kk = [r["k"] for r in rows for _ in r["fp_rows"]]
    P(" FP kind by k:  " + "  ".join(
        f"k={k}: " + ",".join(f"{kd[:8]} {sum(1 for f, kx in zip(fpr, kk) if kx == k and f['kind'] == kd)}" for kd in kinds)
        for k in sorted(set(kk))))
    P(" by pred area (px at 2048): bin  all-preds-in-bin TP-preds  FP  FP-rate  no_gt  match_other  partial_this  partial_other")
    tp_areas = [p[0] * p[1] for r in rows for p in r["tp_pairs"]]
    for lo, hi in zip(abins[:-1], abins[1:]):
        ff = [f for f in fpr if lo <= f["area"] < hi]
        nt = sum(lo <= a < hi for a in tp_areas)
        P(f"  [{lo:>6.0f},{hi:>8.0f})  TP {nt:5d}  FP {len(ff):5d}  prec {nt / max(nt + len(ff), 1):.3f}  "
          + "  ".join(f"{sum(f['kind'] == kd for f in ff):4d}" for kd in kinds))
    P(" by peak prob: bin  FP  share  no_gt  match_other  partial_this  partial_other")
    for lo, hi in ((0.99, 0.995), (0.995, 0.999), (0.999, 1.01)):
        ff = [f for f in fpr if lo <= f["peak"] < hi]
        P(f"  [{lo:.3f},{min(hi, 1):.3f}]  {len(ff):5d}  {len(ff) / FP:.3f}  "
          + "  ".join(f"{sum(f['kind'] == kd for f in ff):4d}" for kd in kinds))
    P("  (peak values are uint8/255 maps: 0.996 = 254, 1.0 = 255)")

    # 4. boundary
    P("\n== 4. Boundary (TP pairs) ==")
    pr = np.array([p[0] for r in rows for p in r["tp_pairs"]])
    ga = np.array([p[1] for r in rows for p in r["tp_pairs"]])
    ti = np.array([p[2] for r in rows for p in r["tp_pairs"]])
    q = [10, 25, 50, 75, 90]
    P(f" area ratio pred/GT percentiles {dict(zip(q, np.percentile(pr, q).round(3)))}; share pred larger {np.mean(pr > 1):.3f}; "
      f"mean log2 ratio {np.mean(np.log2(pr)):+.3f}")
    for lo, hi in zip(abins[:-1], abins[1:]):
        m = (ga >= lo) & (ga < hi)
        if m.any():
            P(f"  GT area [{lo:>6.0f},{hi:>8.0f}): n {m.sum():5d}  median ratio {np.median(pr[m]):.3f}  "
              f"share larger {np.mean(pr[m] > 1):.3f}  mean IoU {ti[m].mean():.3f}")
    P(" TP IoU percentiles " + str(dict(zip(q, np.percentile(ti, q).round(3)))))
    P(" global per-instance morphology at 2048 (full PQ recomputed, all entries):")
    P(f"  base   PQ {base:.4f}  SQ {S / TP:.4f}  TP {TP} FP {FP} FN {FN}")
    for k in MORPH:
        t = np.concatenate([r["morph"][k][0] for r in rows])
        fp_ = sum(r["morph"][k][1] for r in rows)
        fn_ = sum(r["morph"][k][2] for r in rows)
        v = pq(t.sum(), len(t), fp_, fn_)
        P(f"  {'erode' if k < 0 else 'dilate'} {abs(k)}  PQ {v:.4f} ({v - base:+.4f})  SQ {t.mean():.4f}  TP {len(t)} FP {fp_} FN {fn_}")

    # 5. inter-annotator
    P("\n== 5. Inter-annotator PQ (each annotator as prediction vs each other annotator as GT, same image) ==")
    t = np.concatenate([p[0] for p in pairs])
    fp_ = sum(p[1] for p in pairs)
    fn_ = sum(p[2] for p in pairs)
    ia = pq(t.sum(), len(t), fp_, fn_)
    D2 = len(t) + 0.5 * fp_ + 0.5 * fn_
    P(f" ordered pairs {len(pairs)}  PQ {ia:.4f}  SQ {t.mean():.4f}  RQ {len(t) / D2:.4f}  TP {len(t)} FP {fp_} FN {fn_}")
    rr = [r for r in rows if r["k"] > 1]
    tm = np.concatenate([r["tp_iou"] for r in rr])
    fpm = sum(len(r["fp_rows"]) for r in rr)
    fnm = sum(not g["tp"] for r in rr for g in r["gt_rows"])
    P(f" model on the same multi-annotator entries: PQ {pq(tm.sum(), len(tm), fpm, fnm):.4f}  SQ {tm.mean():.4f}  "
      f"RQ {len(tm) / (len(tm) + 0.5 * fpm + 0.5 * fnm):.4f}")


if __name__ == "__main__":
    main()
