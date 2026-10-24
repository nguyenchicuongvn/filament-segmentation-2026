"""Grid-search instance post-processing on cached val probability maps.

Usage:
  python scripts/tune_post.py --probs outputs/probs/unet512_r34_val --out configs/post_unet512.json
  python scripts/tune_post.py --probs outputs/probs/unet768_r34_val --mode watershed
  python scripts/tune_post.py --probs outputs/probs/<group model>_val --mode group
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.data import encode_masks, load_train, split  # noqa: E402
from filseg.dl import PostParams, instances, load_maps  # noqa: E402
from filseg.metric import PQAccumulator  # noqa: E402

GRIDS = {
    "cc": {
        "thresh": [0.3, 0.4, 0.5, 0.6],
        "min_area": [10, 25, 50, 100],
        "group_radius": [0, 2, 4, 8],
        "out_dilate": [0, 1, 2],
    },
    "watershed": {
        "thresh": [0.3, 0.4, 0.5],
        "seed_thresh": [0.6, 0.7, 0.8, 0.9],
        "min_area": [25, 50, 100],
        "group_radius": [0, 2, 4, 8],
    },
    "group": {
        "thresh": [0.3, 0.4, 0.5],
        "group_thresh": [0.3, 0.4, 0.5, 0.6, 0.7],
        "min_area": [25, 50, 100],
    },
}

_VAL = None


def _val():
    global _VAL
    if _VAL is None:
        _VAL = split(load_train(), "val")
    return _VAL


def _score(args):
    probs_dir, p = args
    acc = PQAccumulator()
    cache: dict[str, list] = {}
    for it in _val():
        if it.image_id not in cache:
            prob, group = load_maps(Path(probs_dir), it.image_id)
            cache[it.image_id] = encode_masks(instances(prob, p, group=group))
        acc.add(it.gt_rles, cache[it.image_id])
    return p, acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", type=Path, required=True)
    ap.add_argument("--mode", choices=list(GRIDS), default="cc")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--grid", help='JSON dict overriding the grid, e.g. \'{"thresh": [0.4, 0.5]}\'')
    args = ap.parse_args()

    grid = json.loads(args.grid) if args.grid else GRIDS[args.mode]
    cands = [PostParams(mode=args.mode, **dict(zip(grid, v))) for v in itertools.product(*grid.values())]
    results = []
    with ProcessPoolExecutor() as pool:
        for p, acc in pool.map(_score, [(str(args.probs), c) for c in cands]):
            results.append((acc.pq, p, acc.summary()))
    results.sort(key=lambda r: -r[0])
    for pq, p, s in results[:10]:
        print(s, p.to_dict())
    default = [s for _, p, s in results if p == PostParams()]
    if default:
        print("default", default[0])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results[0][1].to_dict(), indent=2))
        print("->", args.out)


if __name__ == "__main__":
    main()
