"""Image statistics per GONG station (last letter of stem) for all train + test images, at 1024.
Raw in-disk median / p5 / p95, limb ratio (profile at 0.95R / centre), flat-field std, sharpness
(Laplacian variance of flat image), high-pass noise. Also histogram distance (Wasserstein-1 of the
in-disk raw histogram) between each station's mean histogram. Writes station_stats.csv."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from filseg.classical import fit_disk, radial_profile  # noqa
from filseg.data import load_gray, load_train, test_paths  # noqa

R = 1024


def job(a):
    split, path = a
    g = cv2.resize(load_gray(path), (R, R), interpolation=cv2.INTER_AREA).astype(np.float32)
    cx, cy, dr = fit_disk(g.astype(np.uint8))
    yy, xx = np.mgrid[:R, :R]
    rr = np.hypot(xx - cx, yy - cy)
    ind = rr < dr
    prof = np.maximum(radial_profile(g, rr, dr, ind), 1.0)
    flat = np.where(ind, g / prof, 1.0).astype(np.float32)
    inner = rr < 0.9 * dr
    lap = cv2.Laplacian(flat, cv2.CV_32F)
    hp = flat - cv2.GaussianBlur(flat, (0, 0), 1.0)
    hist = np.bincount(g[inner].astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    hist /= hist.sum()
    return dict(split=split, stem=path.stem, st=path.stem[-2], disk_r=dr, med=float(np.median(g[inner])),
                p5=float(np.percentile(g[inner], 5)), p95=float(np.percentile(g[inner], 95)),
                limb=float(prof[(rr > 0.95 * dr) & ind].mean() / prof[rr < 0.2 * dr].mean()),
                fstd=float(flat[inner].std()), lapv=float(lap[inner].var()) * 1e4, hp=float(hp[inner].std()) * 1e3,
                sat=float((g[ind] >= 254).mean()), hist=hist)


if __name__ == "__main__":
    tr = sorted({t.image_id: t.path for t in load_train()}.items())
    jobs = [("train", p) for _, p in tr] + [("test", p) for p in test_paths()]
    with ProcessPoolExecutor(4) as ex:
        rows = list(ex.map(job, jobs, chunksize=8))
    import pandas as pd
    hists = {}
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "hist"} for r in rows])
    df.to_csv(Path(__file__).parent / "station_stats.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.groupby(["st", "split"])[["disk_r", "med", "p5", "p95", "limb", "fstd", "lapv", "hp", "sat"]].median().round(3))
    print("\nwithin-station SD (train):")
    print(df[df.split == "train"].groupby("st")[["med", "fstd", "lapv", "hp"]].std().round(3))
    # Wasserstein-1 between mean in-disk histograms per station (train), in grey levels
    H = {}
    for r in rows:
        if r["split"] == "train":
            H.setdefault(r["st"], []).append(r["hist"])
    cdf = {s: np.cumsum(np.mean(v, 0)) for s, v in H.items()}
    sts = sorted(cdf)
    print("\nW1 between station mean histograms (grey levels):")
    print("   " + " ".join(f"{s:>6}" for s in sts))
    for a in sts:
        print(a, " " + " ".join(f"{np.abs(cdf[a] - cdf[b]).sum():6.1f}" for b in sts))
    # per-image W1 to own station mean vs to global mean
    glob = np.cumsum(np.mean([r["hist"] for r in rows if r["split"] == "train"], 0))
    w_own = [np.abs(np.cumsum(r["hist"]) - cdf[r["st"]]).sum() for r in rows if r["split"] == "train"]
    w_glob = [np.abs(np.cumsum(r["hist"]) - glob).sum() for r in rows if r["split"] == "train"]
    print(f"\nper-image W1 to own-station mean: median {np.median(w_own):.1f}; to global mean: {np.median(w_glob):.1f}")
