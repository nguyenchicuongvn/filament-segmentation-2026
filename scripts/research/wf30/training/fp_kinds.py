"""Classify each (annotator-entry, prediction) FP and each FN into kinds, and test whether map
confidence separates hit vs none-matched predictions. Usage: fp_kinds.py <probs> <post.json> <tag>"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2, numpy as np
from pycocotools import mask as mu
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from filseg.data import by_image, load_train, with_maps
from filseg.dl import PostParams, instances_lowres, load_maps
PROBS = Path(sys.argv[1]); POST = PostParams(**json.loads(Path(sys.argv[2]).read_text())); TAG = sys.argv[3]

def iou(a, b):
    if not a or not b: return np.zeros((len(a), len(b)))
    return np.asarray(mu.iou(a, b, [0] * len(b)))

def job(args):
    stem, entries = args
    prob, group = load_maps(PROBS, stem)
    inst = instances_lowres(prob, POST, group); n = int(inst.max())
    up = cv2.resize(inst.astype(np.float32), (2048, 2048), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    pred = [mu.encode(np.asfortranarray((up == k).astype(np.uint8))) for k in range(1, n + 1)]
    feats = []
    for k in range(1, n + 1):
        pv = prob[inst == k]
        feats.append((pv.mean(), np.mean(pv > 0.99), np.percentile(pv, 10), len(pv)))
    fps, fns = [], []
    hitany = np.zeros(n, bool)
    for key, gts in entries:
        m = iou(gts, pred)  # gt x pred
        if m.size == 0:
            fps += [(len(entries), 'nogt', 0.0, j) for j in range(n)]
            fns += [(len(entries), 'nopred', 0.0, 0) for _ in gts]
            continue
        hit = m > 0.5
        hitany |= hit.any(0)
        inter = m > 0.05
        for j in np.where(~hit.any(0))[0]:
            b = m[:, j].max()
            touched = np.where(inter[:, j])[0]
            if b < 0.05: kind = 'isolated'
            elif len(touched) >= 2: kind = 'merge'          # pred spans >=2 GTs
            elif hit[touched[0]].any(): kind = 'fragment'   # its GT is matched by another pred
            elif b > 0.3: kind = 'nearmiss'
            else: kind = 'partial'
            fps.append((len(entries), kind, b, j))
        for i in np.where(~hit.any(1))[0]:
            b = m[i].max(); touched = np.where(inter[i])[0]
            if b < 0.05: kind = 'missed'
            elif any(inter[:, j].sum() >= 2 for j in touched): kind = 'merged'
            elif len(touched) >= 2: kind = 'split'
            elif b > 0.3: kind = 'nearmiss'
            else: kind = 'partial'
            fns.append((len(entries), kind, b, mu.area(gts[i])))
    return fps, fns, [(f + (bool(hitany[k]),)) for k, f in enumerate(feats)]

def main():
    items = load_train(); val = with_maps(items, PROBS)
    groups = [(s, [(t.key, t.gt_rles) for t in es]) for s, es in by_image(val).items()]
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(job, groups, chunksize=4))
    from collections import Counter
    fps = [f for r in res for f in r[0]]; fns = [f for r in res for f in r[1]]; feats = np.array([f for r in res for f in r[2]], float)
    print(f"== {TAG}: FP entries {len(fps)}, FN entries {len(fns)}")
    for name, L in [("FP", fps), ("FN", fns)]:
        c = Counter(k for _, k, _, _ in L)
        print(f"  {name} kinds: " + ", ".join(f"{k} {v} ({v/len(L):.2f})" for k, v in c.most_common()))
        for n in (1, 2, 3):
            c = Counter(k for a, k, _, _ in L if a == n)
            print(f"    n_ann={n}: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    from sklearn.metrics import roc_auc_score
    y = feats[:, 4]
    for c, nm in enumerate(["meanp", "frac>0.99", "p10"]):
        print(f"  AUC hit-vs-none by {nm}: {roc_auc_score(y, feats[:, c]):.3f}")
    print(f"  frac>0.99 median: hit {np.median(feats[y==1,1]):.2f} none {np.median(feats[y==0,1]):.2f}")
    np.savez(Path(__file__).parent / f"kinds_{TAG}.npz", feats=feats)

if __name__ == '__main__':
    main()
