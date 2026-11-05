# Research 05/11: instance segmentation for thin, elongated objects (agent report, verbatim summary)

Repos cloned read-only to scratchpad (not run): omnipose, Skeleton-Recall, cbDice, clDice,
SpatialEmbeddings, elf. Nothing copied into the project.

## Licences
| Repo | Licence | OSI? |
|---|---|---|
| omnipose | Omnipose NonCommercial License (UW) | No: reimplement from the paper only |
| SpatialEmbeddings (Neven 2019) | CC BY-NC 4.0 | No (its lovasz_losses.py is Berman's, MIT header) |
| Skeleton-Recall (MIC-DKFZ) | Apache-2.0 | Yes |
| cbDice | Apache-2.0 (+monai) | Yes |
| clDice | MIT | Yes |
| elf (mutex watershed) | MIT (bioimage-cpp backend not checked) | Yes |

## Caps from existing evidence
- F31: 59% of near-miss predictions are a TP for another annotator.
- spine_group_oracle.txt: perfect fragment grouping is worth at most +0.023.
- 1 px erode/dilate at 2048 costs -0.014 / -0.029, so boundary work must be sub-pixel.
- Expect +0.003..0.01 per idea at best.

## Methods
- **Omnipose** (Cutler 2022): regresses an eikonal distance field plus flows; the masks come from
  hysteresis on the distance and Euler integration of pixels onto the skeleton.
  - Code: core/flows.py:445-526, core/fields.py:176-234, core/loss.py:13-80, core/masks.py:357-448.
  - Targets: merges, and possibly the size bias.
  - Port: a 1–3 channel head, written clean-room (non-OSI licence).
- **StarDist / Cellpose:** wrong shape prior for curved filaments. Skip.
- **Mutex watershed** (elf mutex_watershed.py:149-185): short-range attractive plus long-range
  repulsive affinities.
  - Port: an affinity head whose target is the fraction of annotators that put both pixels in the
    same instance, then MWS on the hysteresis-fragment graph (about 40 lines of union-find).
  - Ceiling +0.023. Overlaps the spine-grouping work.
- **Spatial embeddings** (Neven 2019): the useful part is the **per-instance Lovász hinge**
  (lovasz_losses.py:80-116, MIT). Our JML1 used a global Lovász, which large filaments dominate.
- **Skeleton Recall** (Kirchhoff ECCV 2024, Apache):
  - Target: a 2 px tube skeleton (skeletonization.py:20-44). Loss: -(sum p*tube+1)/(sum tube+1),
    recall only (dice.py:9-60), weight 1 on top of CE+Dice.
  - Paper: DRIVE beta0 57.0 -> 38.75, Dice +0.1; overhead +8% time.
  - Targets splits and thin parts. Risk: small masks grow.
- **cbDice, clDice, Topograph/Betti matching, boundary/Hausdorff losses, road centreline+radius:**
  already rejected (the clDice family) or a poor fit.

## Ranked experiments (gate: @512 weight sweep first; full runs need Δ >= +0.005, CI > 0)
1. **Skeleton Recall loss** on the main map.
   - Soft tube = mean of per-annotator tubes; w in {0.1, 1}.
   - Also check gt_split and the small-filament area ratio.
2. **Signed-distance head** (Omnipose-style, clean-room) with level-set mask extraction.
   - Variant A: d > tau replaces the probability threshold.
   - Variant B: probability seeds plus d boundaries.
   - Speculative.
3. **Long-range affinity head + fragment-graph mutex watershed**, replacing the radius grouping.
4. **Per-instance Lovász hinge** on bbox+16 px crops, w in {0.1, 0.3}.
5. **(free once 2 or 3 exist)** Add regressed-field features to the GBM rescorer.

Sources: arXiv 2404.03010 (Skeleton Recall); Nature Methods 2022 doi 10.1038/s41592-022-01639-4 (Omnipose).
