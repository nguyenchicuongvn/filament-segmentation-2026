"""Descriptive stats of the `spine` / `bbox` / `area` fields of the Kaggle train JSON.

Read-only on data/MAGFiLO_1.0_Kaggle_2026. Usage: .venv/bin/python scripts/research/spine_stats.py
"""
import json
from collections import Counter

import cv2
import numpy as np
from pycocotools import mask as mu

from filseg.data import TRAIN_ANN, H, W

c = json.loads(TRAIN_ANN.read_text())
anns = c["annotations"]
types = Counter(type(a["spine"]).__name__ for a in anns)
nested = sum(1 for a in anns if a["spine"] and isinstance(a["spine"][0], list))
npts = np.array([len(a["spine"]) // 2 if not (a["spine"] and isinstance(a["spine"][0], list)) else -1 for a in anns])
print("spine types", types, "nested", nested)
print("points per spine: min %d p10 %d med %d p90 %d max %d; empty %d; <2 pts %d" % (
    npts.min(), *np.percentile(npts, [10, 50, 90]).astype(int), npts.max(), (npts == 0).sum(), (npts < 2).sum()))
npoly = np.array([len(a["segmentation"]) for a in anns])
print("polygons per segmentation:", Counter(npoly.tolist()).most_common(6))

rows = []
for a in anns:
    s = np.asarray(a["spine"], float).reshape(-1, 2)
    rle = mu.merge(mu.frPyObjects(a["segmentation"], H, W))
    m = mu.decode(rle)
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad = 8
    y0p, x0p = max(0, y0 - pad), max(0, x0 - pad)
    crop = m[y0p:y1 + pad, x0p:x1 + pad]
    ncc, _ = cv2.connectedComponents(crop.astype(np.uint8), connectivity=8)
    length = float(np.sqrt((np.diff(s, axis=0) ** 2).sum(1)).sum()) if len(s) > 1 else 0.0
    # fraction of spine samples inside the mask
    if len(s) > 1:
        seg = []
        for p, q in zip(s[:-1], s[1:]):
            n = max(2, int(np.ceil(np.hypot(*(q - p)))) + 1)
            seg.append(np.linspace(p, q, n))
        pts = np.concatenate(seg)
    else:
        pts = s
    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
    inside = m[yi, xi].mean() if len(pts) else np.nan
    # does the spine connect separate mask components?
    lab = cv2.connectedComponents(crop.astype(np.uint8), connectivity=8)[1]
    hit = set(lab[np.clip(yi - y0p, 0, crop.shape[0] - 1), np.clip(xi - x0p, 0, crop.shape[1] - 1)].tolist()) - {0}
    bx, by, bw, bh = a["bbox"]
    # spine bbox vs ann bbox
    rows.append(dict(area=a["area"], marea=int(m.sum()), ncc=ncc - 1, length=length, inside=inside,
                     ncc_hit=len(hit), npts=len(s), bw=bw, bh=bh,
                     sx0=s[:, 0].min() - bx, sy0=s[:, 1].min() - by,
                     sx1=bx + bw - s[:, 0].max(), sy1=by + bh - s[:, 1].max(),
                     mx0=x0 - bx, my0=y0 - by, cat=a["category_id"]))

import pandas as pd
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.describe(percentiles=[.01, .1, .5, .9, .99]).T.to_string())
print("area field == decoded mask area: frac |diff|<=1%:", (abs(df.area - df.marea) <= 0.01 * df.marea).mean())
print("instances whose GT mask has >1 connected component:", (df.ncc > 1).sum(), "/", len(df))
print("  ... of those, spine touches >=2 components:", ((df.ncc > 1) & (df.ncc_hit >= 2)).sum())
print("spine fully inside mask (>=0.99):", (df.inside >= .99).mean(), " >=0.9:", (df.inside >= .9).mean(),
      " <0.5:", (df.inside < .5).mean())
print("length/sqrt(area) median", np.median(df.length / np.sqrt(df.marea)))
el = df.length / (df.marea / np.maximum(df.length, 1))
print("elongation length/width p10/50/90", np.percentile(el, [10, 50, 90]))
df.to_csv("outputs/spine_stats.csv", index=False)
