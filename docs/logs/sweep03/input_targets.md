# Sweep 03/11 — input_targets (agent report, verbatim)

# Input and target side: measurements and ranked ideas (E9a holdout, 213 images, hysteresis config)

Every script and output is in `scripts/research/sweep03/sweep_input/`. No tracked repo file was edited.

**Headline.** One input-side effect is real, it comes from the model, and the docs do not have it yet: **the model loses a lot of PQ on sharp-seeing images, and humans do not.**
- Model, E9a holdout: PQ 0.445 → 0.409 → 0.374 across terciles of image high-frequency energy.
- Humans (annotator vs annotator, 296 multi-annotator train images): 0.348 → 0.339 → 0.334.

Everything else on the input side measured as no gain or no shift: filter channels, histogram matching, limb correction, blur/noise augmentation. The input/target levers that are left are small: annotator-count sampling, a 1.125 scale TTA, and skeleton/core targets.

## (a) Measurements

**M1. Instance-level separability of image filters** (`collect_feats.py` → `feats.pkl`; `analyze_feats.py` → `analyze_feats.out`)
- Setup: 1,379 predicted instances. Label "good" means mean matched IoU over the image's annotators > PQ/2 = 0.205 (the F26 rule); 69.8% are good.
- Image features, all computed on the radially flattened plus median-flat-fielded 1024 image: flat intensity, contrast vs a ring around the instance, black-hat r7, DoG(1.5, 8), Sato black ridges with σ 1–3, and Sato inside/ring ratio.
- Best single image feature is the Sato ratio, AUC 0.631. Contrast gives 0.601. For comparison, pmean alone gives 0.729.
- 5-fold-by-image GBM, 3 seeds:

| Features | CV AUC |
|---|---|
| Map features only | 0.6973 ± 0.005 |
| Filters only | 0.6606 |
| Map + filters | 0.6996 |

- Adding filters gains **+0.002 AUC**. Among the 1,374 confident predictions (pmax ≥ 0.995) the filters' AUC stays 0.53–0.63.
- Conclusion: vesselness, top-hat and DoG carry no information the network does not already use. F34 shows an AUC gain of +0.01 already gives ΔPQ ≈ 0, so these features will not help as a filter or as input channels.

**M2. GT faintness** (same output)
- Missed GTs are less dark than detected ones within each size bin. AUC(detected | darker) is 0.72 for < 800 px, 0.72 for 800–3,200 px, 0.83 for > 3,200 px.
- Lone GTs (multi-annotator images, drawn by only one annotator) are also fainter: median contrast 0.909 vs 0.881 for supported GTs, AUC 0.70.
- Faintness and annotator disagreement are the same population. This matches F29/F30; contrast enhancement would mostly add matches to only one annotator.

**M3. Station statistics, all 707 train + 180 test images** (`station_stats.py` → `station_stats.out`, `station_stats.csv`)
- Images are already normalised: in-disk median is 130–133 at every station, limb ratio 0.61–0.63, disk radius 452–455 px at 1024.
- Wasserstein-1 distance between station mean histograms is 0.4–1.6 grey levels. T is the exception at 3.5–5.1.
- Per-image W1 to its own station mean is 1.9, and to the global mean 2.0. Station explains almost none of the histogram variation.
- Test matches train per station on every statistic. T has the highest fine-structure energy (hp 30.6 train / 27.6 test) and its within-station spread is 2× the others.

**M4. PQ per station, E9a holdout**
- B 0.423, C 0.374, L 0.389, M 0.427, T 0.419, U 0.434. Bootstrap SD is 0.015–0.027 each, so this is mostly noise.
- The annotator-group effect is larger, for example C: groups 01/04 0.339 vs other groups 0.410.
- C's low score comes from its many sharp images (20 of 40 C images are in the top sharpness tercile, see M5).

**M5 (new finding). Seeing sharpness vs PQ** (`sharp.py`, `human_by_sharp.py`, `hp_buckets.py` → `.out` files of the same names; example crops in `hp_crops.png`)
- Sharpness measure: Laplacian variance (or high-pass SD; the two correlate at ρ = 0.98) of the flattened disk. The crops show high values mean good seeing with dense dark fibrils, not noise.
- Model E9a PQ by tercile: 0.447 / 0.412 / 0.377. Low minus high = **0.070, 95% CI [0.035, 0.111]**.
- The drop holds in both annotator strata: groups 01/04 0.421 → 0.347, other groups 0.477 → 0.412.
- Humans on the same terciles (train multi-annotator images): 0.349 / 0.341 / 0.330, a drop of only 0.014–0.019. The gap is the model's, not label ambiguity.
- In the sharp tercile:
  - FN/entry is 3.58 vs 2.25 and FP/entry 2.66 vs 1.97.
  - 40% of GTs are < 800 px, vs 26% in the smooth tercile.
  - Small-GT recall is 0.42 vs 0.47.
  - **Recall of filaments that another annotator also drew falls from 0.83 to 0.70**, while recall of lone filaments stays flat at about 0.30.
- So the sharp tercile concentrates the model-fixable "missed despite consensus" FN bucket (F29, 6.5% overall).
- Caveat: model and human numbers come from different image sets (213 holdout vs 296 multi-annotator train), the same caveat F30 carries.

**M6. TTA and perturbation probe on MPS, first 62 holdout images** (`tta_scale.py`, `eval_tta.py` → `eval_tta.out`)
- I stopped it at 62 images: the machine is overloaded (load average ~300), and 213 images would have taken about 95 minutes.
- Fixed post config tuned on the base maps, so the comparison is biased toward base. The base map reproduces the cached maps (max diff 2/255).

| Variant | ΔPQ vs base | 95% CI |
|---|---|---|
| Input ×1.125 (1152) alone | +0.0004 | [−0.009, +0.010] |
| Base + 1152 average | **+0.0028** | [−0.0025, +0.0082] |
| Input ×0.875 (896) | **−0.016** | [−0.028, −0.003] |
| Base + 896 + 1152 | −0.006 | |
| Input blurred σ 1.5 px @1024 ("bad seeing") | **−0.096** | [−0.12, −0.07] |
| Base + blur | −0.040 | |

- The model depends heavily on fine detail, which again points at resolution. Upscaled TTA is roughly neutral to slightly positive; downscaling hurts.

**M7. Metric weight vs training sample weight**
- With one sample per image (N1 `SoftDataset`), 3-annotator images make up **21% of samples but 39% of metric entries** (and 37% of GTs). 1-annotator images are 58% of samples vs 36% of entries.
- On the holdout, model PQ by annotator count is about equal (0.414 / 0.408 / 0.408).
- 3-annotator images hold 32% of the PQ denominator.

## (b) Ranked ideas

| # | Idea | Mechanism and error bucket | External evidence | Expected PQ (conservative) | GPU | Effort | Cheap check |
|---|---|---|---|---|---|---|---|
| 1 | **Make sharpness tercile a G1 diagnostic for N2 (crop 1024 @2048); weight the N2 decision toward it** | M5: −0.07 model-only gap on sharp images, consensus-filament recall 0.83 → 0.70 (F29 "missed despite other annotators", F12). M6: blurring costs −0.096, so the model uses fine detail that is lost at 1024. F14: small-object resolution ceiling 0.860 → 0.881 | S10a/S15 native resolution; S13 against | If N2 fixes half of the sharp-tercile gap, about +0.01 overall. No gain is possible if it does not | 0 extra (N2 already queued) | Tiny: tercile split in `compare.py` output, terciles from `station_stats.csv` | Yes: once `effb4_c2048_f0` arrives, rerun `hp_buckets.py` on its maps. Prediction: ΔPQ is concentrated in tercile 2 and in supported-GT recall. If N2's gain is uniform across terciles, suspect seed noise |
| 2 | **Sample images ∝ number of annotators** (`WeightedRandomSampler` in `SoftDataset`; or loss weight = n_ann) | M7: metric weights 3-annotator images 1.9× what training sees; their soft targets are the only place the network learns where consensus ends (F26/F15) | umerkang66 does exactly this, no ablation (REPORT, "sampled in proportion to annotation sets") | +0.000 to +0.004, not measurable alone (F24) | 0 if folded into the next planned runs (v2 fold 1 or the remaining folds) | 3 lines | No. Fold it into a package; do not ablate alone |
| 3 | **Scale TTA: average D4 at 1024 and at 1152** (no downscale) | Reduces boundary variance; slight TP/FP shift (1152 alone: TP +34, FP +76) | umerkang66: multi-scale TTA +0.0006, not adopted. biconcavelens: multi-scale YOLO TTA +0.02 val but LB tie | +0.001 (M6: +0.0028, CI includes 0) | Inference only, 2× test time; no training | Small (`predict_prob` option) | Yes: finish `tta_scale.py base,s1152` on all 213 images (~30 min of MPS when the Mac is idle), cross-tune with `compare.py`; redo on N1 OOF maps. Adopt only if CI > 0 |
| 4 | **Skeleton + core (distance-transform) auxiliary heads, targets derived from the soft masks** (mask-derived, so the rules allow it; no spine labels) | Too-thin near-misses (F28 18% of FP, F29 14.5% of FN, pred/GT area 0.58) and cut-in-two (F29 6%): a core/skeleton head forces one connected spine per filament | S11a: blob + skeleton/core + deep supervision + JML1 = +0.0092 (90% CI [+0.006, +0.012]); N3 has only the blob part | +0.003 on top of v2 | 2 fold runs ≈ 2.6 h (only if v2 passes G2) | Medium: targets via `skimage.morphology.skeletonize` / `distanceTransform` on the soft target, 2 extra output channels | No |
| 5 | **Oversample sharp-seeing images** (weight by hp tercile) | M5 gap | None | 0 to +0.004; risks shifting FP/FN balance on smooth images | Bundle only | 3 lines | Only after #1: if N2 does not close the gap, this is the cheap fallback |
| 6 | **Native-resolution detail channel for the 1024 model**: 2nd input = 2048 high-pass local energy pooled to 1024 | M5/M6: gives the 1024 network sub-pixel texture without 4× compute | None (M1 shows Sato/DoG at 1024 add nothing, but that is at 1024 and is a different signal) | Unknown, ≤ +0.005. Only worth it if N2 wins but is too expensive for 5 folds | 1–2 runs | Medium (`in_channels=2`, ImageNet weight hack; test path changes) | Partial: per-instance AUC of the pooled 2048 high-pass (extend `collect_feats.py`). Not run |

Nothing else on the input/target side reaches the bar. I would stop at 4 real items (1 to 4), with 5 and 6 conditional on N2's result.

## (c) Rejected, with reasons

- **Multi-channel stacks (flat-field / radial / top-hat / black-hat / DoG / Frangi-Sato) as input**: M1 gives +0.002 instance AUC over map features (0.6973 → 0.6996); F34 shows +0.01 AUC gives ΔPQ ≈ 0. The vesselness-fusion literature (arXiv 2402.14509) is 3D CT, not this task; EdgeAttNet (arXiv 2509.02964, MAGFiLO) has no preprocessing ablation.
- **Limb-darkening correction, flat-fielding, polar or heliographic reprojection**: F28 (3/794 FP from limb or position, 1/2,194 predictions beyond ±70°). M3: limb ratio is already 0.61–0.63 everywhere. biconcavelens tried 8 flat-field and unsharp variants on FITS: no gain.
- **Local contrast normalisation / CLAHE**: S9e, −0.006 to −0.013, monotonic in clip limit. M2: the extra faint structures it would bring out are mostly lone-annotator filaments (AUC 0.70).
- **Log/gamma transforms, histogram matching across the 6 stations**: M3, station histograms differ by ≤ 1.6 grey levels (T ≤ 5); within-station variation equals between-station. Test matches train per station. Gamma jitter is already in the augmentation.
- **Photometric augmentation (per-station style, seeing blur, noise, JPEG)**: no train/test shift to cover (M3). M6 shows blur destroys information the model uses; teaching blur invariance works against M5. biconcavelens: full dihedral + photometric augmentation on Mask R-CNN tied (−0.003 to +0.001, 4/5 seeds negative).
- **Scale / shear / elastic augmentation as a gain lever**: disk radius is constant (452–455 px @1024, train = test), so there is no scale shift to cover. Elastic warps change exactly the filament outlines the metric scores. No evidence. Bundle at most (umerkang66 uses scale 0.87–1.15, no ablation).
- **Copy-paste / cutmix of filaments**: biconcavelens, −0.015 (0.4152 → 0.3998). It also pastes one annotator's style out of its annotator context.
- **Downscale or multi-scale (896) TTA**: M6, −0.016 [−0.028, −0.003].
- **Oversampling hard annotator groups 01/04**: test annotators are unknown (S7). It would shift the model toward one style; F11's gap is style, not image difficulty (M4: within station, group 01/04 is always lower). Same reasoning as the judge's rejection of the style head.
- **Per-annotator-style targets or heads**: S7, and judge score 1.5 (RESEARCH § 10.3).
- **Boundary label smoothing**: the errors run in opposite directions (small masks too large F13, large masks too thin F29), and symmetric smoothing cannot fix both. Soft targets already smooth boundaries on multi-annotator images. No evidence.
- **Special handling of lone filaments**: the soft target already gives them 1/3 or 1/2. Under F26 a lone filament in a 3-annotator image has mean IoU ≈ IoU/3 ≈ 0.22, right at the PQ/2 threshold, so threshold 0.5 on a calibrated soft map already approximates the optimal rule. Nothing left to gain.
- **Centre/offset or affinity instance targets**: aimed at the merge/cut buckets (F28 5.4%, F29 14.2% + 6%). The closest thing tried, the group-map target E8, gave +0.003 (not distinguishable). Rules-based splitting traded one error for the other (F23). A real instance model is N4's job; a pixel-level proxy is not worth ~2.6 GPU-h.
- **Crop sampling centred on filaments**: already implemented (`fg_frac` 0.7 in `SoftDataset` crop mode).
- **Class balance tweaks**: no measured bucket points at it. FP/TP is already 0.57; more positive weight raises FP (F18).

## Caveats

- M6 covers 62 images with a config tuned on the base maps. It ranks the TTA variants; it does not size their gain.
- M5 terciles come from one model (E9a, hard targets, per-annotator sampling); recheck on the N1 OOF maps when they land (`hp_buckets.py` needs only `tables.pkl`-style inputs, so rerun `collect_feats.py` with `PROBS` pointed at the new maps).
- Sources outside RESEARCH.md that I used: umerkang66 REPORT (sampling ∝ annotation sets; multi-scale TTA +0.0006; scale augmentation 0.87–1.15), https://github.com/umerkang66/solar-filament-segmentation/blob/master/REPORT.md; biconcavelens RESULTS.md (copy-paste −0.015; photometric augmentation tie; FITS flat-field variants negative), https://github.com/biconcavelens/filament-segmentation-2026/blob/master/RESULTS.md.
