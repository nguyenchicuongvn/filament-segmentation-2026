"""Ceiling of predicting at a lower resolution: downsample each GT annotation to res, upsample with
the pipeline's upsample_labels and measure IoU with the original annotation: the boundary cost of
predicting at that resolution with an otherwise perfect model.

Usage: python scripts/res_ceiling.py"""
import sys
from pathlib import Path
import numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor
from pycocotools import mask as mu
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filseg.data import load_train, split
from filseg.dl import upsample_labels

def one(args):
    t, res = args
    out = []
    lab = np.zeros((res, res), np.int32)
    for k, r in enumerate(t.gt_rles, 1):
        m = cv2.resize(mu.decode(r).astype(np.float32), (res, res), interpolation=cv2.INTER_AREA) > 0.5
        lab[m & (lab == 0)] = k
    up = upsample_labels(lab) if res != 2048 else lab
    for k, g in enumerate(t.gt_rles, 1):
        p = up == k
        v = mu.iou([mu.encode(np.asfortranarray(p.astype(np.uint8)))], [g], [0])[0][0] if p.any() else 0.0
        out.append((v, mu.area(g)))
    return out

if __name__ == '__main__':
    items = [t for t in split(load_train(), "val") if t.gt_rles]
    with ProcessPoolExecutor() as ex:
        for res in (512, 1024, 1536):
            r = np.array([x for o in ex.map(one, [(t, res) for t in items]) for x in o])
            ious, area = r[:, 0], r[:, 1]
            s = area < 800
            print(res, f"mean IoU {ious.mean():.3f} frac>0.5 {np.mean(ious>0.5):.3f} | <800px: meanIoU {ious[s].mean():.3f} frac>0.5 {np.mean(ious[s]>0.5):.3f}", flush=True)
