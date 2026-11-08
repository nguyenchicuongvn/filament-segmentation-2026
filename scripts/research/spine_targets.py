"""Prototype target for an auxiliary spine head (option a): per image, fraction of annotators whose spine
passes each pixel, spines drawn as polylines of thickness `thick` px at the working resolution and
blurred by a Gaussian (sigma px). Prints the positive-pixel fraction (to set the head's loss weight /
pos_weight) and how much of the mask soft target the spine target covers.
Usage: .venv/bin/python scripts/research/spine_targets.py [res=1024] [n_images=60]"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from filseg.data import TRAIN_ANN, by_image, load_train
from filseg.dl import soft_cached


def spine_soft(entries, spines_by_key, res, thick=3, sigma=1.0):
    acc = np.zeros((res, res), np.float32)
    s = res / 2048
    for it in entries:
        m = np.zeros((res, res), np.uint8)
        for sp in spines_by_key.get(it.key, []):
            pts = np.round(np.asarray(sp, float).reshape(-1, 2) * s).astype(np.int32)
            cv2.polylines(m, [pts], False, 1, thick)
        acc += m
    t = acc / len(entries)
    if sigma:
        t = cv2.GaussianBlur(t, (0, 0), sigma)
        t = np.clip(t / max(t.max(), 1e-6) * (acc.max() / len(entries)), 0, 1) if t.max() > 0 else t
    return t


if __name__ == "__main__":
    res = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    coco = json.loads(TRAIN_ANN.read_text())
    spines = {}
    for a in coco["annotations"]:
        spines.setdefault(a["image_id"], []).append(a["spine"])
    groups = list(by_image(load_train()).values())[:n]
    pos, fgm, cover = [], [], []
    for g in groups:
        t = spine_soft(g, spines, res)
        _, soft = soft_cached(g, res)
        pos.append((t > 0.05).mean())
        fgm.append((soft > 0).mean())
        cover.append(((t > 0.05) & (soft == 0)).sum() / max((t > 0.05).sum(), 1))
    print(f"res {res}, {n} images: spine-target positive fraction {np.mean(pos)*1e3:.2f} per mille; "
          f"mask soft-target positive {np.mean(fgm)*1e3:.2f} per mille; spine px outside mask target {np.mean(cover):.3%}")
