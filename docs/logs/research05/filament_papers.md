# Research 05/11: solar filament papers and their open code (agent report, condensed)

Repos cloned read-only to scratchpad (not run): dasjar/EdgeAttNet, GF-Zhu/Filament-Unet,
adiercke/DeepFilamentSegmentation, irisaltHu/MORDEN, human-com-h/solar-filament-structure,
HELIO-HFC/TrackFil. Nothing copied into the project.

**Bottom line.** No filament paper reports PQ, and none separates instances better than heuristics
on a semantic mask. Weights trained on MAGFiLO v1.0 are off-limits for us (EdgeAttNet,
Hao & Li 2026).

## Papers
| Paper | Method / data | Result | Instance, barbs, small filaments | Code / licence |
|---|---|---|---|---|
| Zhu 2019, Sol. Phys. 294:117, arXiv 1909.06580 | U-Net + dropout, BCE, 512; BBSO/HSOS (~60 pairs) | DSC 0.894 | semantic; per-image ROC threshold; small filaments worse | GF-Zhu/Filament-Unet, MIT |
| Ahmadzadeh 2019, arXiv 1912.02743 | Mask R-CNN, HEK labels, BBSO | "competes" with HEK | Mask R-CNN instances | none |
| Liu 2021, Sol. Phys. 296:176 | U-Net+ASPP, cGAN; HSOS+BBSO | – | semantic | none |
| Guo 2022, Sol. Phys. 297:104 | CondInst + Matrix NMS; BBSO ~10k filaments | full text not read | the only real per-pixel instance model | none |
| Zheng/Hao 2024, arXiv 2402.14209 | U-Net + CSRT tracking + graph spine; CHASE | qualitative | spine via shortest path | none |
| Diercke 2024, A&A 686 A213, arXiv 2402.15407 | YOLOv5 boxes, then semi-supervised U-Net on GONG; BCE pos_weight 10 | 92% accuracy | instances via boxes; misses small filaments | adiercke/DeepFilamentSegmentation, **GPL-3.0** (ideas only) |
| Reche & Cid 2025, Acta Astronautica | DETR + U-Net + tracker; GONG | claims SOTA | via boxes | none |
| Zhu/Lin 2025 "Flat U-Net", ApJ 980:176, arXiv 2502.07259 | light U-Net with channel attention; BBSO | DSC 0.82 | semantic | Zenodo, MIT |
| Solomon 2025 "EdgeAttNet", arXiv 2509.02964 | edge map into bottleneck MHSA; **MAGFiLO** | mIoU 0.645 vs 0.572 | aims at barbs | dasjar/EdgeAttNet, MIT; **model code missing** from the repo, weights on MAGFiLO v1.0 |
| Chalmers & Ahmadzadeh 2025, arXiv 2509.18214 | chirality classification | 0.69/0.73 | uses chirality labels (not allowed) | – |
| Hu 2026 "MORDEN", RAA, arXiv 2607.24525 | multi-head U-Net, focal loss, **DenseCRF**, hole fill, **DBSCAN** for instances; CHASE | IoU 0.732 vs 0.670 | experts: 84% of clusters correct, "fragment integration suboptimal" | irisaltHu/MORDEN, MIT |
| Hao & Li 2026 (structural supervision) | 8 methods × 3 seeds on MAGFiLO @1024: BCE-Dice, Skeleton Recall, component-balanced recall, clDice, Flat U-Net | **no consistent gain** from skeleton/component losses | spine via skeletonization | human-com-h/solar-filament-structure, MIT; weights on MAGFiLO v1.0 (not allowed) |
| Hao 2013 (Sol. Phys. 286:385), 2015 (ApJS 221:33) | classical: disk fit, polynomial-surface brightness flattening, modified Otsu | catalogue | **merge fragments when endpoint distance is small AND tilt angles are similar** (stops neighbouring active-region filaments from merging) | MATLAB, not public |
| Bernasconi 2005 (SFDT); Fuller 2005 (HFC) | threshold + region growing, distance join, spine by thinning | – | – | HELIO-HFC/TrackFil, MIT (tracking only) |

No SAM/foundation-model filament paper was found.

## Code worth borrowing (ideas)
- **Diercke** `dfs/data/data_set.py:182-255`: limb darkening as a degree-6 polynomial of intensity
  vs cos(r·π/2), a 36-term Zernike background fit, a radial residual profile, clip [0.8, 1.3].
  - Pseudo-mask rule `:67-80`: in a box, `I <= mean - std`.
- **MORDEN** `post_processing/tools/densecrf.py:7-36`: pydensecrf (MIT). It binarizes first; we
  would pass the soft map. Also `dbscan.py:7-48` (eps 25, minPts 70 @1K) and
  `hole_filling.py:5-15`.
- **solar-filament-structure** `frozen_sources/mechanism/data_pipeline.py:65`: the Skeleton
  Recall target `dilation(skeletonize(mask), diamond(2)) & mask`; loss weight 0.05.

## Ranked ideas (agent)
1. **Endpoint + orientation-aware fragment grouping** (Hao 2013/2015). Post-processing only.
   Targets merges and splits.
2. **Size-conditional threshold/extent per instance.** Post-processing; targets near-misses.
3. **Intensity-guided boundary refinement** (in-box `mean - k·std`, or per-instance DenseCRF on the
   soft map). Post-processing; weak evidence.
4. **Limb-darkening / background-flattened input channel.** Retrain needed; weak to moderate
   evidence.
5. **Down-up resize blur and gamma augmentation.** Retrain; weak evidence.
6. **"Gap" class between touching instances** (nuclei-style). Retrain + watershed; speculative.
7. **Skeleton Recall / component-balanced loss.** Evidence is negative (Hao & Li 2026).

**Not recommended:**
- EdgeAttNet: code missing, weights on MAGFiLO v1.0.
- Self-training on the GONG archive: likely overlaps test dates.
- Any MAGFiLO-trained public weights.
