"""Candidate component trees on E9a holdout maps, with exact per-annotator IoU at 2048.
Output: one pickle per image with nodes (features, parent, level, family) and IoU vs every GT."""
import sys, pickle, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from pycocotools import mask as mu
from filseg.data import load_train, with_maps, load_gray
from filseg.dl import PostParams, instances_lowres, load_maps, _disk
from filseg.instfilter import instance_features

PROBS = ROOT / "outputs/colab/unet1024_effb4/probs_val"
OUT = Path(__file__).parent / "cands"
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
FAMS = {"r0": 0, "r4": 4}
MIN_AREA = 50

def labels_at(prob, t, r):
    fg = (prob > t).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    keep = st[:, cv2.CC_STAT_AREA] >= 1; keep[0] = False
    if r > 0:
        g = cv2.dilate(fg, _disk(r))
        _, glab = cv2.connectedComponents(g, connectivity=8)
        lab = np.where(fg > 0, glab, 0).astype(np.int32)
    # renumber & drop small
    areas = np.bincount(lab.ravel())
    k = areas >= MIN_AREA; k[0] = False
    lut = np.zeros(len(areas), np.int32); lut[k] = np.arange(1, k.sum() + 1)
    return lut[lab]

def gt_sparse(rle):
    m = mu.decode(rle).astype(np.uint16)
    c = m.reshape(1024, 2, 1024, 2).sum((1, 3)).ravel()
    idx = np.nonzero(c)[0]
    return idx, c[idx].astype(np.float64), float(m.sum())

def iou_mat(lab, gts):
    """(n_gt, n_lab) exact IoU at 2048 of nearest-upsampled 1024 labels."""
    n = int(lab.max()); flat = lab.ravel()
    area = np.bincount(flat, minlength=n + 1)[1:] * 4.0
    M = np.zeros((len(gts), n))
    for i, (idx, cnt, ga) in enumerate(gts):
        inter = np.bincount(flat[idx], weights=cnt, minlength=n + 1)[1:]
        M[i] = inter / np.maximum(area + ga - inter, 1)
    return M

def tree_feats(prob, lab, n):
    flat = lab.ravel(); p = prob.ravel()
    cnt = np.bincount(flat, minlength=n + 1)[1:]
    return cnt

def one(args):
    stem, path, ann_rles = args
    out = OUT / f"{stem}.pkl"
    if out.exists():
        return stem
    prob, _ = load_maps(PROBS, stem)
    gray = load_gray(path)
    gts = [[gt_sparse(r) for r in rles] for rles in ann_rles]
    rec = {"stem": stem, "A": len(ann_rles), "n_gt": [len(g) for g in gts], "fams": {}}
    # baseline: strict hysteresis config
    post = PostParams(**json.loads((ROOT / "configs/post_unet1024_effb4_hyst.json").read_text()))
    base = instances_lowres(prob, post)
    rec["base"] = {"ious": [iou_mat(base, g) for g in gts],
                   "feats": instance_features(base, prob, gray)}
    for fam, r in FAMS.items():
        nodes = []
        prev = None
        for li, t in enumerate(LEVELS):
            lab = labels_at(prob, t, r)
            n = int(lab.max())
            if n == 0:
                prev = lab; continue
            feats = instance_features(lab, prob, gray)
            ious = [iou_mat(lab, g) for g in gts]
            flat = lab.ravel()
            area = np.bincount(flat, minlength=n + 1)[1:]
            # parent: label at previous level of any pixel
            if prev is not None and prev.max() > 0:
                par = np.zeros(n + 1, np.int64)
                par[flat] = prev.ravel()  # nested: all pixels share the same parent
                par = par[1:]
            else:
                par = np.zeros(n, np.int64)
            for k in range(n):
                nodes.append(dict(level=li, t=t, k=k + 1, parent_k=int(par[k]), area=int(area[k]),
                                  feats=feats[k], ious=[m[:, k] for m in ious]))
            prev = lab
        rec["fams"][fam] = nodes
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(rec)); tmp.rename(out)
    return stem

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    jobs = {}
    for it in with_maps(load_train(), PROBS):
        jobs.setdefault(it.image_id, (it.path, []))[1].append(it.gt_rles)
    args = [(s, p, g) for s, (p, g) in sorted(jobs.items())]
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else len(args)
    with ProcessPoolExecutor(4) as ex:
        for i, s in enumerate(ex.map(one, args[:lim])):
            if i % 20 == 0: print(i, s, flush=True)
