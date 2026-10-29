import sys
from pathlib import Path

import numpy as np

W = Path(__file__).parent
BINS = [(0, 400), (400, 800), (800, 1600), (1600, 3200), (3200, 8000), (8000, 1e9)]
for tag in sys.argv[1:]:
    z = np.load(W / f"anatomy_{tag}.npz")
    P, G, H = z["P"], z["G"], z["H"]
    # P: nann, hits, area2048, peak, meanp, r/R, mindist(map px), meddist
    print(f"===== {tag}: preds {len(P)}, gts {len(G)}")
    fp0 = P[:, 1] == 0
    fpent = (P[:, 0] - P[:, 1])  # FP entries contributed per pred
    print(f"preds matched by none: {fp0.sum()} ({fp0.mean():.2f}); FP entries total {fpent.sum():.0f}, from none-matched {fpent[fp0].sum():.0f}")
    iso = P[:, 6] > 3  # no GT pixel of any annotator within 3 map px (~6 px @2048)
    print(f"  none-matched preds: isolated (no GT of anyone within 3px@1024) {np.mean(iso[fp0]):.2f}; overlapping/adjacent a GT {np.mean(~iso[fp0]):.2f}")
    print(f"  FP entries from isolated look-alikes: {fpent[fp0 & iso].sum():.0f}; from touching-GT (fragment/shape) {fpent[~iso].sum():.0f}")
    for name, sel in [("hit", ~fp0), ("none", fp0), ("none&iso", fp0 & iso)]:
        q = lambda c: np.percentile(P[sel, c], [25, 50, 75])
        print(f"  {name:9s} n={sel.sum():4d} area p25/50/75 {q(2).round(0)} peak {q(3).round(3)} meanp {q(4).round(2)} r/R {q(5).round(2)}")
    print("  by r/R: frac of preds none-matched")
    for lo, hi in [(0, .5), (.5, .7), (.7, .8), (.8, .9), (.9, 1.2)]:
        sel = (P[:, 5] >= lo) & (P[:, 5] < hi)
        print(f"    [{lo:.1f},{hi:.1f}) n={sel.sum():4d} none {np.mean(fp0[sel]):.2f} iso-none {np.mean((fp0 & iso)[sel]):.2f}")
    print("  by pred area @2048: frac none-matched")
    for lo, hi in BINS:
        sel = (P[:, 2] >= lo) & (P[:, 2] < hi)
        print(f"    [{lo:5.0f},{hi:7.0g}) n={sel.sum():4d} none {np.mean(fp0[sel]):.2f}")
    # G: nann, area, best, maxp, meanp, sup, r/R, grp01
    print("  GT by size: recall | missed GT maxprob p25/50/75 | frac missed with maxprob<0.3 | <0.5 | recall(sup>=1, multi-ann)")
    for lo, hi in BINS:
        sel = (G[:, 1] >= lo) & (G[:, 1] < hi)
        miss = sel & (G[:, 2] <= 0.5)
        ms = sel & (G[:, 0] > 1) & (G[:, 5] >= 1)
        print(f"    [{lo:5.0f},{hi:7.0g}) n={sel.sum():4d} rec {np.mean(G[sel,2]>0.5):.2f} | {np.percentile(G[miss,3],[25,50,75]).round(2)} | "
              f"{np.mean(G[miss,3]<0.3):.2f} | {np.mean(G[miss,3]<0.5):.2f} | {np.mean(G[ms,2]>0.5) if ms.any() else np.nan:.2f} (n={ms.sum()})")
    multi = G[:, 0] > 1
    print("  multi-ann GT support rate by size (frac of GT matched by >=1 other annotator):")
    for lo, hi in BINS:
        sel = multi & (G[:, 1] >= lo) & (G[:, 1] < hi)
        print(f"    [{lo:5.0f},{hi:7.0g}) n={sel.sum():4d} supported {np.mean(G[sel,5]>=1):.2f}")
    if len(H):
        print("  HUMAN (all train multi-ann): recall of annotator A's GT by annotator B, by A's GT size")
        for lo, hi in BINS:
            sel = (H[:, 1] >= lo) & (H[:, 1] < hi)
            print(f"    [{lo:5.0f},{hi:7.0g}) n={sel.sum():5d} human recall {np.mean(H[sel,2]>0.5):.2f} none(<0.1) {np.mean(H[sel,2]<0.1):.2f}")
        print(f"    share of GT < 400 px: human-pairs {np.mean(H[:,1]<400):.3f}")
    # what-if: PQ contribution. FN entries by size
    fn = G[:, 2] <= 0.5
    tot_fn = fn.sum()
    print(f"  FN entries {tot_fn}; share from GT<400 {np.mean(G[fn,1]<400):.2f}, <800 {np.mean(G[fn,1]<800):.2f}; "
          f"share from unsupported multi-ann GT {np.mean((G[fn,0]>1)&(G[fn,5]==0)):.2f}")
