"""Numbers and figures for the technical report (report/), from the out-of-fold maps of final 1.

  python scripts/report_figures.py stats     # docs/logs/final/report_stats.json and rubric_final1.json
  python scripts/report_figures.py figures   # report/figures/*.pdf

The FP filter in configs/ was fitted on all 707 out-of-fold images, so applying it to the same images
would score them in-sample (PQ 0.450 instead of 0.437). `stats` therefore reproduces the filter's
cross-validated scores exactly as scripts/fp_filter.py computes them (same image folds, same model),
keeps instances whose out-of-fold score reaches the saved threshold, checks that PQ equals the
filter's saved cv_pq, and computes from those predictions:
- per annotator entry: sum of matched IoU, TP, FP, FN -> PQ bootstrapped over images (2000 resamples;
  the metric pools over entries, so resampling images and pooling again is the matching bootstrap);
- PQ per training fold (filseg.data.fold_split);
- the host rubric numbers of scripts/rubric_metrics.py (Dice per entry, IoU, 1:n and n:1).
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filseg.data import N_FOLDS, by_image, encode_masks, fold_split, load_gray, load_train, with_maps  # noqa: E402
from filseg.dl import PostParams, instances_lowres, load_maps, upsample_labels  # noqa: E402
from filseg.metric import IOU_THRESHOLD  # noqa: E402

PROBS = ROOT / "outputs/ens/n1n2n9p_oof"
FILTER = ROOT / "configs/fpfilter_ens_n1n2n9p_orient.pkl"
STATS = ROOT / "docs/logs/final/report_stats.json"
KEEP = ROOT / "docs/logs/final/report_keep.json"
RUBRIC = ROOT / "docs/logs/final/rubric_final1.json"
FIG = ROOT / "report/figures"


_MODS: dict = {}


def _script(name: str):
    """scripts/<name>.py as a module (cached; also works in worker processes)."""
    if name in _MODS:
        return _MODS[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _MODS[name] = mod
    return mod


def _table_job(args):
    return _script("fp_filter")._image_table(args)


def predict(stem: str, keep: list[bool]) -> np.ndarray:
    """Final 1 instance labels at 2048 for one OOF image, keeping the instances flagged in `keep`."""
    filt = pickle.loads(FILTER.read_bytes())
    prob, group = load_maps(PROBS, stem)
    inst = instances_lowres(prob, PostParams(**filt["post"]), group)
    lut = np.zeros(len(keep) + 1, np.int32)
    k = np.asarray(keep, bool)
    lut[1:][k] = np.arange(1, int(k.sum()) + 1)
    return upsample_labels(lut[inst])


def cv_keep() -> tuple[dict, dict]:
    """Out-of-fold filter decisions per image, as scripts/fp_filter.py; returns (keep, tables)."""
    fpf = _script("fp_filter")
    filt = pickle.loads(FILTER.read_bytes())
    post = PostParams(**filt["post"])
    by_img: dict[str, tuple[Path, list]] = {}
    for it in with_maps(load_train(), PROBS):
        by_img.setdefault(it.image_id, (it.path, []))[1].append(it.gt_rles)
    jobs = [(str(PROBS), post, p, s, gts) for s, (p, gts) in sorted(by_img.items())]
    with ProcessPoolExecutor(8) as pool:
        tables = list(pool.map(_table_job, jobs, chunksize=4))
    X = np.concatenate([t[1] for t in tables])
    y = np.concatenate([t[2] for t in tables])
    owner = np.repeat(np.arange(len(tables)), [len(t[2]) for t in tables])
    idx = list(range(len(tables)))
    random.Random(0).shuffle(idx)
    fold_of = np.empty(len(tables), int)
    for f, chunk in enumerate(np.array_split(idx, 5)):
        fold_of[chunk] = f
    oof = np.zeros(len(y))
    for f in range(5):
        te = fold_of[owner] == f
        oof[te] = fpf.predict(fpf.make_model(filt["kind"]).fit(X[~te], y[~te]), filt["kind"], X[te])
    keep, start = {}, 0
    for t in tables:
        n = len(t[2])
        keep[t[0]] = oof[start:start + n] >= filt["threshold"]
        start += n
    pq_cv, _ = fpf.pq_with_keep(tables, keep)
    assert abs(pq_cv - filt["cv_pq"]) < 1e-9, (pq_cv, filt["cv_pq"])
    return keep, {t[0]: t for t in tables}


def pq(terms: np.ndarray) -> float:
    s, tp, fp, fn = terms.sum(0)
    return float(s / (tp + 0.5 * fp + 0.5 * fn)) if tp + fp + fn else 0.0


def entry_terms(m: np.ndarray) -> tuple[float, int, int, int]:
    """(sum of TP IoU, TP, FP, FN) from one entry's (n_gt, n_pred) IoU matrix."""
    hit = m > IOU_THRESHOLD
    return float(m[hit].sum()), int(hit.sum()), int((hit.sum(0) == 0).sum()), int((hit.sum(1) == 0).sum())


def rubric_job(args) -> list[dict]:
    stem, keep, entries = args
    rm = _script("rubric_metrics")
    labels = predict(stem, keep)
    preds = encode_masks(labels)
    union = (labels > 0).astype(np.uint8)
    return [rm.entry_stats(gts, preds, union) for gts in entries]


def stats() -> None:
    keep, tables = cv_keep()
    items = with_maps(load_train(), PROBS)
    stems = sorted(tables)
    per_entry = {s: [entry_terms(m[:, keep[s]]) for m in tables[s][3]] for s in stems}
    img_terms = np.array([np.sum(per_entry[s], 0) for s in stems], float)
    rng = np.random.default_rng(0)
    boot = [pq(img_terms[rng.integers(0, len(stems), len(stems))]) for _ in range(2000)]
    fold_of = {t.image_id: k for k in range(N_FOLDS) for t in fold_split(items, k)[1]}
    folds = {k: pq(np.array([img_terms[i] for i, s in enumerate(stems) if fold_of[s] == k])) for k in range(N_FOLDS)}
    out = {"images": len(stems), "entries": int(sum(len(v) for v in per_entry.values())),
           "pq": pq(img_terms), "pq_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           "pq_boot_sd": float(np.std(boot)), "tp_fp_fn": img_terms[:, 1:].sum(0).astype(int).tolist(),
           "fold_pq": folds, "per_image_pq": {s: pq(np.array(per_entry[s], float)) for s in stems},
           "n_entries": {s: len(per_entry[s]) for s in stems}}
    STATS.parent.mkdir(parents=True, exist_ok=True)
    STATS.write_text(json.dumps(out, indent=1))
    KEEP.write_text(json.dumps({s: keep[s].tolist() for s in stems}))
    print(json.dumps({k: out[k] for k in ("images", "entries", "pq", "pq_ci95", "pq_boot_sd", "tp_fp_fn", "fold_pq")},
                     indent=1), flush=True)

    by = by_image(items)
    jobs = [(s, keep[s].tolist(), [t.gt_rles for t in by[s]]) for s in stems]
    with ProcessPoolExecutor(8) as pool:
        rows = [r for rs in pool.map(rubric_job, jobs, chunksize=4) for r in rs]
    rm = _script("rubric_metrics")
    tp = out["tp_fp_fn"][0]
    summary = {"probs": str(PROBS), "filter": f"{FILTER} (out-of-fold scores)", "images": len(stems),
               "pq": out["pq"], "sq": float(img_terms[:, 0].sum() / tp), "tp": tp, "fp": out["tp_fp_fn"][1],
               "fn": out["tp_fp_fn"][2], **rm.summarise(rows)}
    summary["rq"] = summary["pq"] / summary["sq"]
    RUBRIC.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("pq", "sq", "rq", "dice_mean", "tp_iou_mean", "one_n_rate",
                                              "n_one_rate")}, indent=1))


# ---------------------------------------------------------------- figures
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "font.family": "sans-serif", "axes.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    return plt


def fig_pipeline(plt) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    boxes = [
        ("H-$\\alpha$ image\n2048$^2$\ngray / 255", "#eeeeee"),
        ("15 U-Nets\nN1 B4 @1024\nN2 B4 crops\nN9$'$ V2-S crops", "#dbe8f5"),
        ("8$\\times$ TTA\nfold mean\nfamilies 1:1:1", "#dbe8f5"),
        ("hysteresis\n$p > 0.5$\npeak $> 0.99$\narea $\\geq$ 200", "#f5e6d3"),
        ("oriented\ngrouping\nr 6, $\\pm$16 px", "#f5e6d3"),
        ("FP filter\nboosted trees\n17 features", "#f5e6d3"),
        ("COCO RLE\nno overlaps", "#eeeeee"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 0.95))
    ax.set_xlim(0, len(boxes))
    ax.set_ylim(0, 1)
    ax.axis("off")
    w = 0.86
    for i, (txt, col) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((i + 0.07, 0.1), w, 0.8, boxstyle="round,pad=0.01,rounding_size=0.06",
                                    fc=col, ec="#555555", lw=0.6))
        ax.text(i + 0.07 + w / 2, 0.5, txt, ha="center", va="center", fontsize=6, linespacing=1.2)
        if i:
            ax.add_patch(FancyArrowPatch((i - 0.07, 0.5), (i + 0.07, 0.5), arrowstyle="-|>", mutation_scale=6,
                                         lw=0.6, color="#333333"))
    fig.savefig(FIG / "pipeline.pdf", bbox_inches="tight", pad_inches=0.01)


def _crop_box(masks: list[np.ndarray], size: int = 900) -> tuple[int, int, int]:
    union = np.zeros(masks[0].shape, bool)
    for m in masks:
        union |= m > 0
    ys, xs = np.nonzero(union)
    if not len(ys):
        return 574, 574, size
    cy, cx = int(np.median(ys)), int(np.median(xs))
    y0 = int(np.clip(cy - size // 2, 0, 2048 - size))
    x0 = int(np.clip(cx - size // 2, 0, 2048 - size))
    return y0, x0, size


def _overlay(ax, gray: np.ndarray, labels: np.ndarray, box, title: str, fill: bool) -> None:
    import cv2
    y0, x0, s = box
    g = gray[y0:y0 + s, x0:x0 + s]
    lab = labels[y0:y0 + s, x0:x0 + s]
    rgb = np.repeat(g[..., None], 3, 2).astype(np.float32) / 255
    cmap = np.array([[0.90, 0.25, 0.20], [0.20, 0.55, 0.95], [0.95, 0.75, 0.10], [0.30, 0.80, 0.35],
                     [0.80, 0.35, 0.85], [0.10, 0.80, 0.80]])
    for j, k in enumerate(np.unique(lab[lab > 0])):
        m = (lab == k).astype(np.uint8)
        c = cmap[j % len(cmap)]
        if fill:
            rgb[m > 0] = 0.55 * rgb[m > 0] + 0.45 * c
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        edge = np.zeros_like(m)
        cv2.drawContours(edge, cnts, -1, 1, 3)
        rgb[edge > 0] = c
    ax.imshow(np.clip(rgb, 0, 1), interpolation="lanczos")
    ax.set_title(title, fontsize=6.5, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig_examples(plt, st: dict) -> None:
    """Two out-of-fold images with 2+ annotators: H-alpha crop, each annotator, final 1 prediction."""
    items = with_maps(load_train(), PROBS)
    by = by_image(items)
    keep = json.loads(KEEP.read_text())
    multi = sorted((s for s in by if len(by[s]) >= 2 and st["per_image_pq"][s] > 0),
                   key=lambda s: st["per_image_pq"][s])
    picks = [multi[int(len(multi) * 0.75)], multi[int(len(multi) * 0.35)]]
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.55))
    for row, stem in zip(axes, picks):
        es = by[stem][:2]
        gray = load_gray(es[0].path)
        gts = []
        for e in es:
            lab = np.zeros((2048, 2048), np.int32)
            for i, r in enumerate(e.gt_rles):
                lab[mask_util.decode(r) > 0] = i + 1
            gts.append(lab)
        pred = predict(stem, keep[stem])
        box = _crop_box(gts + [pred])
        y0, x0, s = box
        row[0].imshow(gray[y0:y0 + s, x0:x0 + s], cmap="gray", interpolation="lanczos")
        row[0].set_title(f"{stem} (crop)", fontsize=6.5, pad=2)
        row[0].set_xticks([])
        row[0].set_yticks([])
        for sp in row[0].spines.values():
            sp.set_visible(False)
        for j, (e, lab) in enumerate(zip(es, gts)):
            _overlay(row[1 + j], gray, lab, box, f"annotator {chr(65 + j)} ({len(e.gt_rles)} filaments)", False)
        _overlay(row[3], gray, pred, box, f"final 1 (image PQ {st['per_image_pq'][stem]:.2f})", True)
    fig.tight_layout(pad=0.2, w_pad=0.3, h_pad=0.8)
    fig.savefig(FIG / "examples.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)


def fig_rubric(plt) -> None:
    r = json.loads(RUBRIC.read_text())
    fig, ax = plt.subplots(1, 3, figsize=(3.4, 1.25))
    edges = np.linspace(0, 1, 11)
    ax[0].bar(edges[:-1], r["dice_hist"], width=0.1, align="edge", color="#4c72b0", ec="white", lw=0.4)
    ax[0].set_title(f"Dice per entry\n(mean {r['dice_mean']:.3f})", fontsize=6.5)
    ax[1].bar(edges[:-1], r["gt_best_iou_hist"], width=0.1, align="edge", color="#55a868", ec="white", lw=0.4)
    ax[1].axvline(0.5, color="k", ls="--", lw=0.6)
    ax[1].set_title("best IoU per GT\nfilament", fontsize=6.5)
    x = np.arange(4)
    c1 = [r["one_n_counts"][str(k)] for k in range(4)]
    c2 = [r["n_one_counts"][str(k)] for k in range(4)]
    ax[2].bar(x - 0.2, c1, 0.4, color="#c44e52", label="pred. per GT (1:n)")
    ax[2].bar(x + 0.2, c2, 0.4, color="#8172b2", label="GT per pred. (n:1)")
    ax[2].set_xticks(x, ["0", "1", "2", "3+"])
    ax[2].set_title(f"1:n {100 * r['one_n_rate']:.1f}%, n:1 {100 * r['n_one_rate']:.1f}%", fontsize=6.5)
    ax[2].set_ylim(0, 1.45 * max(c1 + c2))
    ax[2].legend(fontsize=4.8, frameon=False, loc="upper right", handlelength=1)
    for a in ax:
        a.tick_params(labelsize=5.5, length=2, width=0.5)
    fig.tight_layout(pad=0.2, w_pad=0.4)
    fig.savefig(FIG / "rubric.pdf", bbox_inches="tight", pad_inches=0.01)


def figures() -> None:
    plt = _style()
    FIG.mkdir(parents=True, exist_ok=True)
    st = json.loads(STATS.read_text())
    fig_pipeline(plt)
    fig_rubric(plt)
    fig_examples(plt, st)
    print("wrote", sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    {"stats": stats, "figures": figures}[sys.argv[1]]()
