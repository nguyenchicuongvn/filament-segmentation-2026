"""SHA256 of every cached map in some probability-map dirs, to check a rerun reproduces them.

  python scripts/map_hashes.py --dirs n1=outputs/colab/effb4_s1024/probs_test ... --out hashes.json
  python scripts/map_hashes.py --compare a.json b.json      # per dir: identical maps, and the worst one
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", help="name=dir")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", nargs=2, type=Path)
    ap.add_argument("--maps", nargs=2, type=Path, help="with --compare: two map dirs to diff pixel-wise")
    a = ap.parse_args()
    if a.compare:
        x, y = (json.loads(p.read_text()) for p in a.compare)
        for name in sorted(set(x) & set(y)):
            same = sum(x[name].get(s) == h for s, h in y[name].items())
            print(f"{name:12s} {same}/{len(y[name])} maps identical")
        if a.maps:
            worst, diff_px, total = 0, 0, 0
            for f in sorted(f for f in a.maps[0].glob("*.npy") if not f.name.startswith(".")):
                u = np.load(f).astype(np.int16)
                v = np.load(a.maps[1] / f.name).astype(np.int16)
                d = np.abs(u - v)
                worst, diff_px, total = max(worst, int(d.max())), diff_px + int((d > 0).sum()), total + d.size
            print(f"pixel diff: max {worst}, {diff_px} of {total} pixels differ ({diff_px / total:.2e})")
        return
    out = {}
    for spec in a.dirs:
        name, d = spec.split("=", 1)
        out[name] = {f.stem: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path(d).glob("*.npy"))
                     if not f.name.startswith(".")}  # skip AppleDouble files on exFAT
        print(f"{name}: {len(out[name])} maps")
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
