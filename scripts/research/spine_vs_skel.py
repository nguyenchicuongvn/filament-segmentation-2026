"""How different is the annotated spine from a skeleton computed from the mask (which was always legal)?
Crop-based, sample of annotations. Usage: .venv/bin/python scripts/research/spine_vs_skel.py [n]"""
import json
import sys

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from filseg.data import TRAIN_ANN

n = int(sys.argv[1]) if len(sys.argv) > 1 else 800
anns = json.loads(TRAIN_ANN.read_text())["annotations"]
rng = np.random.default_rng(0)
idx = rng.choice(len(anns), min(n, len(anns)), replace=False)
rows = []
for i in idx:
    a = anns[i]
    s = np.asarray(a["spine"], float).reshape(-1, 2)
    polys = [np.asarray(p, float).reshape(-1, 2) for p in a["segmentation"]]
    allp = np.concatenate(polys + [s])
    x0, y0 = np.floor(allp.min(0)).astype(int) - 6
    x1, y1 = np.ceil(allp.max(0)).astype(int) + 6
    h, w = y1 - y0, x1 - x0
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(p - [x0, y0]).astype(np.int32) for p in polys], 1)
    ncc = cv2.connectedComponents(m, connectivity=8)[0] - 1
    sp = np.zeros((h, w), np.uint8)
    cv2.polylines(sp, [np.round(s - [x0, y0]).astype(np.int32)], False, 1, 1)
    sk = skeletonize(m > 0).astype(np.uint8)
    # distances
    d_to_sk = distance_transform_edt(1 - sk) if sk.any() else np.full((h, w), np.inf)
    d_to_sp = distance_transform_edt(1 - sp)
    d_out = distance_transform_edt(1 - m)  # distance of spine pixels outside the mask
    spl = float(np.sqrt((np.diff(s, axis=0) ** 2).sum(1)).sum())
    width = m.sum() / max(spl, 1)
    # skeleton pixels farther than 1 width from the spine = spurs/barbs
    spur = (sk & (d_to_sp > max(2.0, width))).sum() / max(sk.sum(), 1)
    # fraction of spine pixels within ~width/2 of the skeleton
    cover = (d_to_sk[sp > 0] <= max(2.0, width / 2)).mean()
    outside = (m[sp > 0] == 0).mean()
    # spine length vs mask longest extent (Feret approx via minAreaRect)
    (_, _), (rw, rh), _ = cv2.minAreaRect(np.argwhere(m)[:, ::-1].astype(np.float32))
    rows.append((a["area"], m.sum(), ncc, spl, width, spur, cover, outside, d_out[sp > 0].max(), spl / max(rw, rh, 1), len(s)))
r = np.array(rows, float)
names = ["area", "mask_px", "ncc", "spine_len", "width", "skel_spur_frac", "spine_near_skel", "spine_out_frac",
         "spine_max_out_px", "spine_len/feret", "npts"]
for k, nm in enumerate(names):
    print(f"{nm:18s} p10 {np.percentile(r[:, k], 10):9.3f} med {np.median(r[:, k]):9.3f} p90 {np.percentile(r[:, k], 90):9.3f} mean {r[:, k].mean():9.3f}")
print("masks with >1 component:", int((r[:, 2] > 1).sum()), "/", len(r))
print("spines with >10% outside mask:", int((r[:, 7] > 0.1).sum()), "; max out >= 5px:", int((r[:, 8] >= 5).sum()))
print("skeletons with >20% spur pixels:", int((r[:, 5] > 0.2).sum()))
