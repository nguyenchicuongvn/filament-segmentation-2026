"""Evaluate (and optionally random-search) the classical baseline on train images.

Usage:
  python scripts/eval_classical.py --n 200                       # default params
  python scripts/eval_classical.py --n 150 --search 40 --seed 0  # random search
  python scripts/eval_classical.py --params outputs/best_params.json --split val
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.classical import Params, segment  # noqa: E402
from filseg.data import encode_masks, load_gray, load_train, split  # noqa: E402
from filseg.metric import PQAccumulator  # noqa: E402

SEARCH_SPACE = {
    "bg_kernel": [31, 45, 61, 81, 101],
    "smooth_sigma": [0.0, 1.0, 1.5, 2.5],
    "thresh_high": [0.72, 0.75, 0.78, 0.81, 0.84, 0.87, 0.90],
    "thresh_low": [0.88, 0.90, 0.92, 0.94, 0.96],
    "disk_margin": [0.82, 0.86, 0.90, 0.93, 0.95, 0.97],
    "sunspot_thresh": [0.35, 0.45, 0.55, 0.65],
    "open_radius": [0, 1, 2],
    "close_radius": [0, 2, 3, 5, 7, 9],
    "min_area": [60, 80, 100, 130, 160],
    "group_radius": [0, 4, 8, 12, 18],
    "out_dilate": [0, 1, 2, 3, 4],
}

_TRAIN = None


def _train():
    global _TRAIN
    if _TRAIN is None:
        _TRAIN = load_train()
    return _TRAIN


def _eval_one(args):
    i, p, which, n = args
    item = split(_train(), which)[i]
    pred = encode_masks(segment(load_gray(item.path), p))
    return item.gt_rles, pred


def evaluate(p: Params, which: str, n: int, pool: ProcessPoolExecutor) -> PQAccumulator:
    items = split(_train(), which)
    n = min(n, len(items))
    acc = PQAccumulator()
    for gt, pred in pool.map(_eval_one, [(i, p, which, n) for i in range(n)], chunksize=4):
        acc.add(gt, pred)
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000, help="max images to use")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--params", type=Path, help="JSON file with Params overrides")
    ap.add_argument("--search", type=int, default=0, help="random-search iterations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/best_params.json"))
    args = ap.parse_args()

    base = Params(**json.loads(args.params.read_text())) if args.params else Params()
    with ProcessPoolExecutor() as pool:
        best = evaluate(base, args.split, args.n, pool)
        print("base", best.summary(), flush=True)
        best_p = base
        rng = random.Random(args.seed)
        for it in range(args.search):
            # mutate 1-3 params of the current best (simple local random search)
            cand = best_p
            for k in rng.sample(list(SEARCH_SPACE), rng.randint(1, 3)):
                cand = replace(cand, **{k: rng.choice(SEARCH_SPACE[k])})
            acc = evaluate(cand, args.split, args.n, pool)
            improved = acc.pq > best.pq
            print(f"[{it}] {'*' if improved else ' '} {acc.summary()} {cand.to_dict()}", flush=True)
            if improved:
                best, best_p = acc, cand
    if args.search:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(best_p.to_dict(), indent=2))
        print("best", best.summary(), "->", args.out)


if __name__ == "__main__":
    main()
